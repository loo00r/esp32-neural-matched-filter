//! Fixed-point Q31 requantization helpers and quantization parameter descriptors.
//!
//! All inference operations in `conv1d_int8` use strictly integer arithmetic
//! (`i8`, `u8`, `i32`, `i64`) with zero floating-point instructions in the inner loops.

/// Fixed-point multiplier representing a positive real scale `M = mult * 2^(-shift)`.
///
/// `mult` is stored in Q31 normalized form (`[2^30, 2^31 - 1]`), or `0` for a zero channel.
/// `shift` is in `[1, 62]` so `(acc * mult + rounding) >> shift` never overflows `i64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPointScale {
    pub mult: i32,
    pub shift: u8,
}

impl FixedPointScale {
    pub const ZERO: Self = Self { mult: 0, shift: 31 };

    /// Decomposes a non-negative real scale `real_scale` into `(mult, shift)` such that
    /// `real_scale ~= (mult as f64) / 2^(shift)`.
    ///
    /// Called during offline quantization (host); inference uses only `apply_i32`.
    pub fn from_f64(real_scale: f64) -> Self {
        if !real_scale.is_finite() || real_scale <= 0.0 {
            return Self::ZERO;
        }

        // Normalize into [0.5, 1.0) using binary exponent extraction (no_std friendly).
        let mut mantissa = real_scale;
        let mut exp: i32 = 0;
        while mantissa >= 1.0 {
            mantissa *= 0.5;
            exp += 1;
        }
        while mantissa < 0.5 {
            mantissa *= 2.0;
            exp -= 1;
        }

        // Scale to Q31 integer in [2^30, 2^31].
        let mut q = (mantissa * ((1_u64 << 31) as f64) + 0.5) as i64;
        if q >= (1_i64 << 31) {
            q >>= 1;
            exp += 1;
        }

        let total_shift = 31 - exp;
        if total_shift <= 0 {
            // Clamp extremely large multipliers safely.
            return Self {
                mult: i32::MAX,
                shift: 1,
            };
        }
        if total_shift > 62 {
            return Self::ZERO;
        }

        Self {
            mult: q as i32,
            shift: total_shift as u8,
        }
    }

    /// Computes `round(acc * mult / 2^shift)` using pure integer arithmetic.
    #[inline(always)]
    pub const fn apply_i32(self, acc: i32) -> i32 {
        if self.mult == 0 || acc == 0 {
            return 0;
        }
        let prod = (acc as i64) * (self.mult as i64);
        let shift = self.shift as u32;
        let round = 1_i64 << (shift - 1);
        let scaled = if prod >= 0 {
            (prod + round) >> shift
        } else {
            -(((-prod) + round) >> shift)
        };
        if scaled > i32::MAX as i64 {
            i32::MAX
        } else if scaled < i32::MIN as i64 {
            i32::MIN
        } else {
            scaled as i32
        }
    }

    /// Requantizes an `i32` accumulator with fused ReLU into unsigned `[0, 255]`.
    #[inline(always)]
    pub const fn apply_relu_u8(self, acc: i32) -> u8 {
        if acc <= 0 {
            return 0;
        }
        let val = self.apply_i32(acc);
        if val <= 0 {
            0
        } else if val >= 255 {
            255
        } else {
            val as u8
        }
    }

    /// Requantizes an `i32` accumulator into symmetric signed `[-127, 127]`.
    #[inline(always)]
    pub const fn apply_sym_i8(self, acc: i32) -> i8 {
        let val = self.apply_i32(acc);
        if val <= -127 {
            -127
        } else if val >= 127 {
            127
        } else {
            val as i8
        }
    }
}

/// Quantizes an `f32` frame into symmetric `i8` (`[-127, 127]`) using a static input scale.
#[inline]
pub fn quantize_frame_i8(input: &[f32], scale: f32, output: &mut [i8]) {
    debug_assert_eq!(input.len(), output.len());
    let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        let scaled = src * inv_scale;
        let rounded = if scaled >= 0.0 {
            (scaled + 0.5) as i32
        } else {
            (scaled - 0.5) as i32
        };
        *dst = rounded.clamp(-127, 127) as i8;
    }
}

/// Dequantizes a symmetric `i8` frame back to `f32` using a static output scale.
#[inline]
pub fn dequantize_frame_i8(input: &[i8], scale: f32, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = (src as f32) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_scale_matches_float_across_orders_of_magnitude() {
        let scales = [1e-5_f64, 3.7e-4, 0.0125, 0.3333, 1.0, 4.75, 128.25];
        let inputs = [-100_000_i32, -1234, -1, 0, 1, 1234, 100_000];
        for &s in &scales {
            let fp = FixedPointScale::from_f64(s);
            for &x in &inputs {
                let expected = (x as f64) * s;
                let expected_i32 = if expected >= 0.0 {
                    (expected + 0.5) as i32
                } else {
                    (expected - 0.5) as i32
                };
                let got = fp.apply_i32(x);
                assert!(
                    (got - expected_i32).abs() <= 1,
                    "scale={s}, x={x}: got={got}, expected={expected_i32}"
                );
            }
        }
    }
}
