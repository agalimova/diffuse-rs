#!/usr/bin/env python3
"""GSM8K exact-match evaluation harness for diffuse-rs (DiffusionGemma).

The tokenizer and chat template are reconstructed directly from the GGUF's
embedded SentencePiece (unigram) vocab — the exact tokenizer the model was
trained with. This matters: gemma4 splits multi-digit numbers into single
digits, which a naive longest-match tokenizer gets wrong, wrecking arithmetic.

Usage:
    python gsm8k_eval.py --model dgemma-q4km.gguf \
        --bin ./target/release/diffuse-rs \
        --n 50 --steps 128 --max-tokens 512 --remasking entropy_bound
"""

import argparse
import datetime
import json
import re
import subprocess
import urllib.request

import numpy as np
from gguf import GGUFReader
from tokenizers import Tokenizer, decoders
from tokenizers.models import Unigram
from tokenizers.normalizers import Prepend, Replace
from tokenizers.normalizers import Sequence as NSeq

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)


def load_shots(n):
    """Few-shot prefix from the GSM8K train set: each example's worked solution
    with the final answer phrased as the directive asks ('The answer is N')."""
    if n <= 0:
        return ""
    raw = urllib.request.urlopen(GSM8K_TRAIN_URL, timeout=30).read().decode()
    out = []
    for line in raw.splitlines()[:n]:
        ex = json.loads(line)
        sol, _, num = ex["answer"].partition("####")
        out.append(f"Question: {ex['question']}\n{sol.strip()}\nThe answer is {num.strip().replace(',', '')}.")
    return "\n\n".join(out) + "\n\n"


def build_tokenizer(gguf_path):
    """Reconstruct the gemma4 unigram tokenizer + chat template from the GGUF."""
    reader = GGUFReader(gguf_path)
    fields = reader.fields

    def column(name):
        field = fields[name]
        return [field.parts[d] for d in field.data]

    def scalar(name):
        field = fields[name]
        return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")

    pieces = [bytes(p).decode("utf-8", "replace") for p in column("tokenizer.ggml.tokens")]
    scores = [float(np.array(p).reshape(-1)[0]) for p in column("tokenizer.ggml.scores")]
    types = [int(np.array(p).reshape(-1)[0]) for p in column("tokenizer.ggml.token_type")]

    tokenizer = Tokenizer(Unigram(list(zip(pieces, scores)), unk_id=3, byte_fallback=True))
    # gemma normalizer: prepend a word boundary, then map spaces to it.
    tokenizer.normalizer = NSeq([Prepend("▁"), Replace(" ", "▁")])
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content=" ", left=1, right=0),
        ]
    )
    # CONTROL (3) and USER_DEFINED (4) tokens are matched literally, not encoded.
    specials = [pieces[i] for i, t in enumerate(types) if t in (3, 4)]
    tokenizer.add_special_tokens(specials)

    template = scalar("tokenizer.chat_template")
    return tokenizer, template


# gemma4 states its answer first then explains, so pin the final answer to a
# fixed trailing line that extraction can target unambiguously.
ANSWER_DIRECTIVE = "\nSolve step by step, then end with a line: The answer is <number>."


def render_prompt(template, question, enable_thinking, shots=""):
    """Render the gemma4 chat template for a single user turn, optionally
    prefixed with few-shot worked examples."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    q = (shots + "Question: " + question) if shots else question
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env.from_string(template).render(
        messages=[{"role": "user", "content": q + ANSWER_DIRECTIVE}],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        bos_token="<bos>",
        eos_token="<eos>",
    )


def generate(args, prompt_ids):
    cmd = [
        args.bin, "generate", "--model", args.model,
        "--prompt-ids", ",".join(map(str, prompt_ids)),
        "-n", str(args.max_tokens), "--n-steps", str(args.steps),
        "--threads", str(args.threads), "--remasking", args.remasking,
        "--eb-entropy-bound", str(args.eb_entropy_bound),
    ]
    if args.suppress_ids:
        cmd += ["--suppress-ids", args.suppress_ids]
    if args.no_cache:
        cmd.append("--no-cache")
    # Block decoding: short answers stop at <eos> in the first block (cheap),
    # long answers extend into later blocks up to max-tokens (no truncation),
    # avoiding a flat O(canvas^2)-per-step cost on every problem.
    if args.block_length:
        cmd += ["--block-length", str(args.block_length)]
    if args.eos_id is not None:
        cmd += ["--eos-id", str(args.eos_id)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"Tokens: \[([\d, ]+)\]", result.stdout + result.stderr)
    return [int(x) for x in match.group(1).split(",")] if match else []


def generate_batch_ids(args, prompts):
    """Run several prompts in one batched forward (gen-batch). Amortizes the
    weight read across the batch, so a larger canvas stays affordable.
    Returns one token list per prompt, in order."""
    cmd = [
        args.bin, "gen-batch", "--model", args.model,
        "--prompts", ";".join(",".join(map(str, p)) for p in prompts),
        "-n", str(args.max_tokens), "--n-steps", str(args.steps),
        "--threads", str(args.threads), "--remasking", args.remasking,
        "--eb-entropy-bound", str(args.eb_entropy_bound),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    by_idx = {}
    for m in re.finditer(r"\[(\d+)\] \([^)]*\) Tokens: \[([\d, ]*)\]", text):
        by_idx[int(m.group(1))] = [int(x) for x in m.group(2).split(",") if x.strip()]
    return [by_idx.get(i, []) for i in range(len(prompts))]


def decode_full(tokenizer, ids):
    """Full decoded generation up to the first <eos> (special tokens kept)."""
    return tokenizer.decode(ids, skip_special_tokens=False).split("<eos>")[0]


def _clean(num):
    num = num.replace(",", "").rstrip(".")
    # Normalize integer-valued decimals (gemma writes "57.00" for 57).
    try:
        f = float(num)
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return num


def _extract(text):
    """Pick a number from one text region, most reliable cue first: explicit
    `answer is N`, then a bolded `**N**`, then the last number."""
    explicit = re.findall(r"answer\s*(?:is|:)?\s*\*{0,2}\$?(-?\d[\d,]*\.?\d*)", text, re.I)
    if explicit:
        return _clean(explicit[-1])
    bolded = re.findall(r"\*\*\$?(-?\d[\d,]*\.?\d*)\*{0,2}", text)
    if bolded:
        return _clean(bolded[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _clean(nums[-1]) if nums else None


def final_number(full):
    """Extract the answer from the full generation. gemma4 wraps output in
    channels (`<|channel>thought\\n<channel|>{answer}`), so prefer the default
    channel (after the last `<channel|>`); but fall back to the whole turn if it
    has no number — recovers answers in the thought channel or when the channel
    structure is malformed (which the channel-only cut would discard)."""
    body = full
    for stop in ("<turn|>", "<|turn>"):  # stop at the end of the model turn
        body = body.split(stop)[0]
    regions = []
    if "<channel|>" in body:
        regions.append(body.rsplit("<channel|>", 1)[1])  # default channel (preferred)
    # whole turn, channel markers stripped (not cut) so thought text survives
    regions.append(body.replace("<|channel>", " ").replace("<channel|>", " "))
    for region in regions:
        n = _extract(region)
        if n is not None:
            return n
    return None


def gold_answer(record):
    return record["answer"].split("####")[-1].strip().replace(",", "")


def load_problems(n, start=0):
    raw = urllib.request.urlopen(GSM8K_TEST_URL, timeout=30).read().decode()
    return [json.loads(line) for line in raw.splitlines()[start:start + n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default="./target/release/diffuse-rs")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--threads", type=int, default=18)
    ap.add_argument("--remasking", default="entropy_bound")
    ap.add_argument("--eb-entropy-bound", type=float, default=0.2,
                    help="entropy below which to commit (lower = stricter, fewer per step)")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the inter-step KV cache (exact full recompute each step)")
    ap.add_argument("--suppress-ids", default="",
                    help="comma-separated token ids to forbid (e.g. 105,106 turn markers)")
    ap.add_argument("--start", type=int, default=0,
                    help="problem offset (for splitting across array tasks)")
    ap.add_argument("--block-length", type=int, default=0,
                    help="semi-autoregressive block size (0 = single canvas)")
    ap.add_argument("--eos-id", type=int, default=None,
                    help="stop a block once this token is unmasked (gemma <eos>=1)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot worked examples from the GSM8K train set (0 = zero-shot)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="problems per batched forward (gen-batch); 1 = sequential")
    ap.add_argument("--out", default=None, help="write per-problem results JSON")
    args = ap.parse_args()

    tokenizer, template = build_tokenizer(args.model)
    problems = load_problems(args.n, args.start)
    shots = load_shots(args.shots)
    batch_size = max(1, args.batch_size)

    correct, results, idx = 0, [], args.start
    for start in range(0, len(problems), batch_size):
        chunk = problems[start:start + batch_size]
        prompt_ids = [
            tokenizer.encode(
                render_prompt(template, p["question"], args.thinking, shots), add_special_tokens=False
            ).ids
            for p in chunk
        ]
        outs = (
            [generate(args, prompt_ids[0])]
            if batch_size == 1
            else generate_batch_ids(args, prompt_ids)
        )
        for problem, out_ids in zip(chunk, outs):
            idx += 1
            text = decode_full(tokenizer, out_ids)
            predicted, gold = final_number(text), gold_answer(problem)
            ok = predicted == gold
            correct += ok
            results.append(
                {"idx": idx, "ok": bool(ok), "pred": predicted, "gold": gold, "text": text[:700]}
            )
            print(f"[{idx}/{len(problems)}] {'OK ' if ok else 'XX '} pred={predicted} gold={gold}", flush=True)
            if args.out:
                with open(args.out, "w") as fh:
                    json.dump({"score": correct, "n": idx, "results": results}, fh, indent=2)

    pct = 100 * correct / len(problems)
    print(f"\nGSM8K exact-match: {correct}/{len(problems)} = {pct:.1f}%")


if __name__ == "__main__":
    main()
