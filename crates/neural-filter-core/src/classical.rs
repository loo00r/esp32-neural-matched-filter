//! Classical Matched Filter (CMF) implementations in FP32 and INT8/INT32.
//!
//! Includes both:
//! 1. Synchronized scalar dot-product (`TEMPLATE_LEN` and `FRAME_LEN` MACs) — the
//!    minimal classical baseline referenced in the thesis.
//! 2. Sliding matched filter (`VALID_CORR_LEN` lags x `TEMPLATE_LEN` taps) for
//!    unsynchronized preamble/pulse detection and Time-of-Arrival (ToA) estimation.

use crate::{FRAME_LEN, ResourceFootprint, TEMPLATE_LEN, VALID_CORR_LEN};

/// Detection result produced by sliding matched filtering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionPeak {
    /// Peak matched-filter correlation score (detection test statistic).
    pub peak_value: f32,
    /// Estimated Time-of-Arrival (sample offset in `[0, VALID_CORR_LEN - 1]`).
    pub peak_index: usize,
}

/// Pure scalar dot-product between two equal-length `f32` slices.
#[inline]
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += x * y;
    }
    acc
}

/// Pure integer scalar dot-product between two equal-length `i8` slices with `i32` accumulation.
#[inline]
pub fn dot_product_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0_i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += (x as i32) * (y as i32);
    }
    acc
}

/// Sliding matched filter in `f32` over a full frame of `FRAME_LEN` samples.
///
/// Computes `corr[lag] = sum_{k=0}^{TEMPLATE_LEN-1} frame[lag + k] * template[k]`
/// and returns the maximum correlation peak and estimated Time-of-Arrival.
pub fn sliding_matched_filter_f32(
    frame: &[f32; FRAME_LEN],
    template: &[f32; TEMPLATE_LEN],
    corr_out: &mut [f32; VALID_CORR_LEN],
) -> DetectionPeak {
    let mut peak_value = f32::NEG_INFINITY;
    let mut peak_index = 0_usize;

    for lag in 0..VALID_CORR_LEN {
        let val = dot_product_f32(&frame[lag..lag + TEMPLATE_LEN], template);
        corr_out[lag] = val;
        if val > peak_value {
            peak_value = val;
            peak_index = lag;
        }
    }

    DetectionPeak {
        peak_value,
        peak_index,
    }
}

/// Sliding matched filter in integer `i8 -> i32` arithmetic.
///
/// Returns `(peak_raw_i32, peak_index)`.
pub fn sliding_matched_filter_i8(
    frame: &[i8; FRAME_LEN],
    template: &[i8; TEMPLATE_LEN],
    corr_out: &mut [i32; VALID_CORR_LEN],
) -> (i32, usize) {
    let mut peak_value = i32::MIN;
    let mut peak_index = 0_usize;

    for lag in 0..VALID_CORR_LEN {
        let val = dot_product_i8(&frame[lag..lag + TEMPLATE_LEN], template);
        corr_out[lag] = val;
        if val > peak_value {
            peak_value = val;
            peak_index = lag;
        }
    }

    (peak_value, peak_index)
}

/// Maximum-Likelihood (ML) linear matched-filter waveform reconstruction for a unit-energy template.
///
/// Given peak amplitude `peak_value = <x[tau..tau+K], p>` (where `||p||_2 = 1`),
/// projects the frame onto the detected template subspace: `y_hat[tau + k] = max(peak_value, 0) * p[k]`.
pub fn reconstruct_ml_matched_f32(
    frame: &[f32; FRAME_LEN],
    template: &[f32; TEMPLATE_LEN],
    output: &mut [f32; FRAME_LEN],
) -> DetectionPeak {
    let mut corr = [0.0_f32; VALID_CORR_LEN];
    let peak = sliding_matched_filter_f32(frame, template, &mut corr);

    output.fill(0.0);
    let amp = if peak.peak_value > 0.0 {
        peak.peak_value
    } else {
        0.0
    };
    let dst = &mut output[peak.peak_index..peak.peak_index + TEMPLATE_LEN];
    for (y, &p) in dst.iter_mut().zip(template.iter()) {
        *y = amp * p;
    }
    peak
}

/// Zero-phase linear FIR matched-bandpass output (convolving frame with template, centered).
///
/// Useful as a continuous linear-filter baseline (no hard peak picking).
pub fn linear_fir_matched_f32(
    frame: &[f32; FRAME_LEN],
    template: &[f32; TEMPLATE_LEN],
    output: &mut [f32; FRAME_LEN],
) {
    let half = TEMPLATE_LEN / 2;
    for (t, dst) in output.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for (k, &p) in template.iter().enumerate() {
            let idx = t as isize + k as isize - half as isize;
            if idx >= 0 && (idx as usize) < FRAME_LEN {
                acc += frame[idx as usize] * p;
            }
        }
        *dst = acc;
    }
}

/// Resource footprint of a synchronized single-frame dot product (`FRAME_LEN` taps) in FP32.
pub const fn footprint_dot_product_f32() -> ResourceFootprint {
    ResourceFootprint {
        weight_bytes: FRAME_LEN * core::mem::size_of::<f32>(),
        metadata_bytes: 0,
        peak_activation_bytes: FRAME_LEN * core::mem::size_of::<f32>(),
        macs_per_frame: FRAME_LEN,
    }
}

/// Resource footprint of sliding matched filter (`VALID_CORR_LEN` x `TEMPLATE_LEN`) in FP32.
pub const fn footprint_sliding_mf_f32() -> ResourceFootprint {
    ResourceFootprint {
        weight_bytes: TEMPLATE_LEN * core::mem::size_of::<f32>(),
        metadata_bytes: 0,
        peak_activation_bytes: (FRAME_LEN + VALID_CORR_LEN) * core::mem::size_of::<f32>(),
        macs_per_frame: VALID_CORR_LEN * TEMPLATE_LEN,
    }
}

/// Resource footprint of sliding matched filter (`VALID_CORR_LEN` x `TEMPLATE_LEN`) in INT8.
pub const fn footprint_sliding_mf_i8() -> ResourceFootprint {
    ResourceFootprint {
        weight_bytes: TEMPLATE_LEN,
        metadata_bytes: 8,
        peak_activation_bytes: FRAME_LEN + VALID_CORR_LEN * core::mem::size_of::<i32>(),
        macs_per_frame: VALID_CORR_LEN * TEMPLATE_LEN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_mf_recovers_exact_delay_in_clean_frame() {
        let mut template = [0.0_f32; TEMPLATE_LEN];
        let mut norm_sq = 0.0_f32;
        for (k, v) in template.iter_mut().enumerate() {
            let x = if k % 4 < 2 { 1.0 } else { -1.0 };
            *v = x;
            norm_sq += x * x;
        }
        let inv_norm = 1.0 / (norm_sq as f64).sqrt() as f32;
        for v in &mut template {
            *v *= inv_norm;
        }

        let true_lag = 43;
        let mut frame = [0.0_f32; FRAME_LEN];
        for k in 0..TEMPLATE_LEN {
            frame[true_lag + k] = 1.25 * template[k];
        }

        let mut corr = [0.0_f32; VALID_CORR_LEN];
        let peak = sliding_matched_filter_f32(&frame, &template, &mut corr);
        assert_eq!(peak.peak_index, true_lag);
        assert!((peak.peak_value - 1.25).abs() < 1e-5);
    }
}
