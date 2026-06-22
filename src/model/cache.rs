// =============================================================================
// Inter-step KV cache (approximation: reuse stale K/V for unchanged positions)
// =============================================================================

/// Which positions get recomputed this pass (active) vs served from the
/// cache (cached). Both lists hold original sequence positions, ascending.
pub struct ActiveSplit {
    pub active: Vec<usize>,
    pub cached: Vec<usize>,
}

impl ActiveSplit {
    pub fn all(n: usize) -> Self {
        Self {
            active: (0..n).collect(),
            cached: Vec::new(),
        }
    }
}

/// Positions stay active for this many steps after their token changes. This
/// lets neighbors re-settle before their K/V is frozen into the cache.
pub(crate) const EXTRA_ACTIVE_STEPS: usize = 2;

pub struct StepCache {
    /// Per-layer cached K after RoPE + GQA repeat: [n_layer][n_total * kv_stride]
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Token sequence from the previous step (for change detection)
    prev_seq: Vec<i32>,
    /// Step when each position last changed
    last_changed: Vec<i32>,
    pub initialized: bool,
    /// Per-layer K/V row stride (n_head_kv * head_dim); may differ across
    /// layers for heterogeneous-attention models.
    strides: Vec<usize>,
    /// dInfer-style vicinity refresh: also recompute committed positions within
    /// this many tokens of a still-masked position (they must re-attend to it).
    /// 0 = off (the cache's default aggressive reuse).
    pub vicinity: usize,
}

impl StepCache {
    pub fn new(n_total: usize, strides: Vec<usize>) -> Self {
        Self {
            k: strides.iter().map(|&s| vec![0.0; n_total * s]).collect(),
            v: strides.iter().map(|&s| vec![0.0; n_total * s]).collect(),
            prev_seq: vec![-1; n_total],
            last_changed: vec![-1; n_total],
            initialized: false,
            strides,
            vicinity: 0,
        }
    }

    pub fn update_seq(&mut self, seq: &[i32]) {
        self.prev_seq.copy_from_slice(seq);
        self.initialized = true;
    }

    /// Save fresh K/V rows into the cache at their original positions.
    pub(crate) fn store(&mut self, layer: usize, positions: &[usize], k: &[f32], v: &[f32]) {
        let stride = self.strides[layer];
        for (row, &pos) in positions.iter().enumerate() {
            self.k[layer][pos * stride..(pos + 1) * stride]
                .copy_from_slice(&k[row * stride..(row + 1) * stride]);
            self.v[layer][pos * stride..(pos + 1) * stride]
                .copy_from_slice(&v[row * stride..(row + 1) * stride]);
        }
    }

    /// Gather cached K/V rows into contiguous arrays.
    pub(crate) fn gather(&self, layer: usize, positions: &[usize], out_k: &mut [f32], out_v: &mut [f32]) {
        let stride = self.strides[layer];
        for (row, &pos) in positions.iter().enumerate() {
            out_k[row * stride..(row + 1) * stride]
                .copy_from_slice(&self.k[layer][pos * stride..(pos + 1) * stride]);
            out_v[row * stride..(row + 1) * stride]
                .copy_from_slice(&self.v[layer][pos * stride..(pos + 1) * stride]);
        }
    }

    /// Decide which positions need recomputation this step: changed tokens,
    /// still-masked positions, and recently changed positions.
    pub fn split_active(&mut self, seq: &[i32], is_masked: &[bool], step: usize) -> ActiveSplit {
        let n = seq.len();
        let mut active = vec![false; n];
        for i in 0..n {
            let changed = seq[i] != self.prev_seq[i];
            if changed {
                self.last_changed[i] = step as i32;
            }
            let recently_changed = self.last_changed[i] >= 0
                && (step as i32 - self.last_changed[i]) <= EXTRA_ACTIVE_STEPS as i32;
            active[i] = changed || is_masked[i] || recently_changed;
        }
        // Vicinity refresh: extend the active set to committed positions within
        // `vicinity` of a still-masked position, so they re-attend to it.
        if self.vicinity > 0 {
            for m in (0..n).filter(|&i| is_masked[i]) {
                let (lo, hi) = (m.saturating_sub(self.vicinity), (m + self.vicinity + 1).min(n));
                active[lo..hi].fill(true);
            }
        }
        let mut split = ActiveSplit { active: Vec::new(), cached: Vec::new() };
        for (i, &a) in active.iter().enumerate() {
            if a {
                split.active.push(i);
            } else {
                split.cached.push(i);
            }
        }
        split
    }
}

