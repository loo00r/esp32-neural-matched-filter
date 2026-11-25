//! Adaptive Filter Mode Controller (Classical MF vs Neural INT8 MF) for DVS/DVFS integration.
//!
//! Directly supports the thesis's future-research architecture: switching dynamically
//! between the ultra-low-energy classical matched filter (optimal in clean/AWGN channels)
//! and the INT8 Neural Matched Filter (superior under colored, impulsive, or nonlinear
//! interference), using an O(L) integer-friendly channel non-Gaussianity / coloration detector.

use crate::FRAME_LEN;

/// Operating mode selected by the adaptive controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Clean / near-AWGN channel: use ultra-low-power Classical Matched Filter (can scale CPU to 80 MHz).
    ClassicalLowPower,
    /// Non-AWGN / distorted / colored / impulsive interference: engage INT8 Neural Matched Filter.
    NeuralInt8,
}

/// Lightweight O(L) channel diagnostics computed on a single incoming frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelDiagnostics {
    /// Sample excess kurtosis (`E[(x-mu)^4] / var^2 - 3`). Near 0 for Gaussian; high for impulsive noise,
    /// negative (`< -0.8`) for hard-clipped/saturated nonlinear channels.
    pub excess_kurtosis: f32,
    /// Lag-1 normalized autocorrelation `|R(1) / R(0)|`. Near 0 for white noise; high (`> 0.35`)
    /// for colored industrial noise or narrowband hum.
    pub lag1_autocorr: f32,
}

/// Decision thresholds for adaptive Classical <-> Neural switching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveThresholds {
    /// Switch to neural if `|lag1_autocorr| > max_white_autocorr` (colored / tonal interference).
    pub max_white_autocorr: f32,
    /// Switch to neural if `excess_kurtosis > max_gaussian_kurtosis` (impulsive spikes).
    pub max_gaussian_kurtosis: f32,
    /// Switch to neural if `excess_kurtosis < min_gaussian_kurtosis` (strong saturation / clipping).
    pub min_gaussian_kurtosis: f32,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self {
            max_white_autocorr: 0.32,
            max_gaussian_kurtosis: 1.5,
            min_gaussian_kurtosis: -0.75,
        }
    }
}

/// Computes `O(FRAME_LEN)` channel diagnostics (zero allocations, `no_std`).
pub fn analyze_channel(frame: &[f32; FRAME_LEN]) -> ChannelDiagnostics {
    let n = FRAME_LEN as f32;
    let mut sum = 0.0_f32;
    for &x in frame {
        sum += x;
    }
    let mean = sum / n;

    let mut m2 = 0.0_f32;
    let mut m4 = 0.0_f32;
    let mut lag1 = 0.0_f32;
    let mut prev = frame[0] - mean;
    for (i, &x) in frame.iter().enumerate() {
        let d = x - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
        if i > 0 {
            lag1 += d * prev;
        }
        prev = d;
    }

    let var = m2 / n;
    if var < 1e-12 {
        return ChannelDiagnostics {
            excess_kurtosis: 0.0,
            lag1_autocorr: 0.0,
        };
    }

    let excess_kurtosis = (m4 / n) / (var * var) - 3.0;
    let lag1_autocorr = (lag1 / m2).abs();
    ChannelDiagnostics {
        excess_kurtosis,
        lag1_autocorr,
    }
}

/// Selects between `FilterMode::ClassicalLowPower` and `FilterMode::NeuralInt8`.
pub fn select_filter_mode(
    frame: &[f32; FRAME_LEN],
    thresholds: AdaptiveThresholds,
) -> (FilterMode, ChannelDiagnostics) {
    let diag = analyze_channel(frame);
    let non_awgn = diag.lag1_autocorr > thresholds.max_white_autocorr
        || diag.excess_kurtosis > thresholds.max_gaussian_kurtosis
        || diag.excess_kurtosis < thresholds.min_gaussian_kurtosis;
    let mode = if non_awgn {
        FilterMode::NeuralInt8
    } else {
        FilterMode::ClassicalLowPower
    };
    (mode, diag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_impulsive_spike_as_neural_mode() {
        let mut frame = [0.0_f32; FRAME_LEN];
        for (i, x) in frame.iter_mut().enumerate() {
            *x = if i % 2 == 0 { 0.1 } else { -0.1 };
        }
        frame[50] = 6.0; // heavy industrial switching impulse
        let (mode, diag) = select_filter_mode(&frame, AdaptiveThresholds::default());
        assert_eq!(mode, FilterMode::NeuralInt8);
        assert!(diag.excess_kurtosis > 10.0);
    }

    #[test]
    fn detects_strong_colored_drift_as_neural_mode() {
        let mut frame = [0.0_f32; FRAME_LEN];
        for (i, x) in frame.iter_mut().enumerate() {
            *x = (i as f32 * 0.08).sin();
        }
        let (mode, diag) = select_filter_mode(&frame, AdaptiveThresholds::default());
        assert_eq!(mode, FilterMode::NeuralInt8);
        assert!(diag.lag1_autocorr > 0.9);
    }
}
