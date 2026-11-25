//! Deterministic, zero-dependency PRNG (`Xoshiro256**` seeded via SplitMix64)
//! for 100% reproducible signal synthesis, training, and benchmarking.

use std::f64::consts::TAU;

#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn from_seed(mut seed: u64) -> Self {
        let mut state = [0_u64; 4];
        for slot in &mut state {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = z ^ (z >> 31);
        }
        Self {
            state,
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.state[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform float in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Uniform float in `[low, high)`.
    #[inline]
    pub fn uniform_f32(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }

    /// Uniform integer in `[low, high_exclusive)`.
    #[inline]
    pub fn range_usize(&mut self, low: usize, high_exclusive: usize) -> usize {
        debug_assert!(low < high_exclusive);
        let span = (high_exclusive - low) as u64;
        low + (self.next_u64() % span) as usize
    }

    /// Standard Gaussian sample `N(0, 1)` via Box-Muller transform.
    #[inline]
    pub fn normal_f32(&mut self) -> f32 {
        if let Some(z) = self.spare_normal.take() {
            return z as f32;
        }
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        self.spare_normal = Some(r * theta.sin());
        (r * theta.cos()) as f32
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range_usize(0, i + 1);
            slice.swap(i, j);
        }
    }
}
