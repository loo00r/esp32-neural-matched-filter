//! IoT baseband signal template (LFM chirp) and realistic low-SNR physical channel models.
//!
//! Uses standard RF/DSP per-sample pulse SNR (`SNR_pulse = 10 * log10(P_pulse / P_noise)`),
//! where a 32-sample chirp yields `10 * log10(32) = +15.05 dB` classical matched-filter
//! processing gain in AWGN. Operating in `[-16, +2] dB` mirrors real low-SNR IoT receivers
//! (LoRa / CSS / sub-GHz industrial telemetry) where detection is genuinely challenging.

use std::f64::consts::TAU;

use neural_filter_core::{FRAME_LEN, TEMPLATE_LEN};

use crate::rng::Rng;

/// Signal propagation and interference regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelRegime {
    /// Additive White Gaussian Noise (classical matched filter is Neyman-Pearson optimal).
    Awgn,
    /// AR(1) colored industrial background noise + strong in-band narrowband tonal interference.
    Colored,
    /// Nonlinear RF/ADC hardware distortion (saturation + harmonic intermodulation) + colored noise.
    NonlinearColored,
    /// Full harsh industrial IoT channel: colored AR(1) + narrowband hum + heavy-tailed
    /// switching impulses (Middleton-like) + nonlinear PA/multipath signal distortion.
    IndustrialHarsh,
}

impl ChannelRegime {
    pub const ALL: [Self; 4] = [
        Self::Awgn,
        Self::Colored,
        Self::NonlinearColored,
        Self::IndustrialHarsh,
    ];

    pub const fn slug(self) -> &'static str {
        match self {
            Self::Awgn => "awgn",
            Self::Colored => "colored",
            Self::NonlinearColored => "nonlinear_colored",
            Self::IndustrialHarsh => "industrial_harsh",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Awgn => "AWGN (Baseline)",
            Self::Colored => "Colored + Narrowband",
            Self::NonlinearColored => "Nonlinear + Colored",
            Self::IndustrialHarsh => "Industrial Harsh (Non-AWGN)",
        }
    }
}

/// One labeled baseband frame for training or evaluation.
#[derive(Debug, Clone)]
pub struct FrameSample {
    pub input: [f32; FRAME_LEN],
    pub target: [f32; FRAME_LEN],
    pub true_lag: Option<usize>,
    pub snr_db: f32,
}

/// Generates the canonical IoT Linear Frequency Modulated (LFM) chirp preamble template.
///
/// Properties:
/// - Hann-windowed chirp sweeping normalized frequency `0.08 -> 0.34` cycles/sample (`BT ~= 8.3`).
/// - Strictly zero-mean (`sum p[k] = 0`) and unit L2 energy (`||p||_2 = 1.0`).
pub fn build_lfm_chirp_template() -> [f32; TEMPLATE_LEN] {
    let f0 = 0.08_f64;
    let f1 = 0.34_f64;
    let n = TEMPLATE_LEN as f64;
    let mut raw = [0.0_f64; TEMPLATE_LEN];

    for (k, slot) in raw.iter_mut().enumerate() {
        let kf = k as f64;
        let window = 0.5 * (1.0 - (TAU * kf / (n - 1.0)).cos());
        let phase = TAU * (f0 * kf + 0.5 * (f1 - f0) * kf * kf / n);
        *slot = window * phase.cos();
    }

    let mean = raw.iter().sum::<f64>() / n;
    for v in &mut raw {
        *v -= mean;
    }

    let norm = raw.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut out = [0.0_f32; TEMPLATE_LEN];
    for (dst, &src) in out.iter_mut().zip(raw.iter()) {
        *dst = (src / norm) as f32;
    }
    out
}

const LAG_MARGIN: usize = 8;

/// Generates one synthetic frame under `regime` at per-sample pulse SNR `snr_pulse_db`.
///
/// Specifically:
/// - Signal average power over active pulse support is `P_s = A^2 / TEMPLATE_LEN`.
/// - Total noise power per sample is `P_n = P_s * 10^(-snr_pulse_db / 10)`.
/// - Frame noise energy across `FRAME_LEN` samples is `E_n = FRAME_LEN * P_n`.
pub fn generate_sample(
    regime: ChannelRegime,
    template: &[f32; TEMPLATE_LEN],
    snr_pulse_db: f32,
    has_signal: bool,
    rng: &mut Rng,
) -> FrameSample {
    let mut target = [0.0_f32; FRAME_LEN];
    let mut tx_sig = [0.0_f32; FRAME_LEN];

    let (amp, true_lag) = if has_signal {
        let amp = rng.uniform_f32(0.80, 1.25);
        let lag = rng.range_usize(LAG_MARGIN, FRAME_LEN - TEMPLATE_LEN - LAG_MARGIN + 1);
        for k in 0..TEMPLATE_LEN {
            target[lag + k] = amp * template[k];
            tx_sig[lag + k] = target[lag + k];
        }
        // In NonlinearColored and IndustrialHarsh, transmitter PA / multipath distorts the emitted waveform
        if matches!(
            regime,
            ChannelRegime::NonlinearColored | ChannelRegime::IndustrialHarsh
        ) {
            // Mild 2-tap multipath echo + nonlinear PA compression on transmitted pulse
            let echo_gain = rng.uniform_f32(0.25, 0.45);
            let echo_delay = 3_usize;
            for k in (0..TEMPLATE_LEN).rev() {
                let idx = lag + k + echo_delay;
                if idx < FRAME_LEN {
                    tx_sig[idx] += echo_gain * target[lag + k];
                }
            }
        }
        (amp, Some(lag))
    } else {
        (1.0_f32, None)
    };

    // Per-sample pulse power and corresponding total frame noise energy
    let p_signal = (amp as f64) * (amp as f64) / (TEMPLATE_LEN as f64);
    let p_noise_sample = p_signal / 10.0_f64.powf(snr_pulse_db as f64 / 10.0);
    let desired_noise_energy = p_noise_sample * (FRAME_LEN as f64);

    let noise = synthesize_noise(regime, desired_noise_energy, rng);

    let mut input = [0.0_f32; FRAME_LEN];
    match regime {
        ChannelRegime::Awgn | ChannelRegime::Colored => {
            for t in 0..FRAME_LEN {
                input[t] = tx_sig[t] + noise[t];
            }
        }
        ChannelRegime::NonlinearColored => {
            // AGC-normalized nonlinear RF/ADC saturation + 2nd/3rd harmonic intermodulation.
            // Scaling saturation knee relative to RMS keeps realistic distortion across all SNRs.
            let mut mixed = [0.0_f32; FRAME_LEN];
            let mut rms_sq = 0.0_f64;
            for t in 0..FRAME_LEN {
                let z = tx_sig[t] + noise[t];
                mixed[t] = z;
                rms_sq += (z as f64) * (z as f64);
            }
            let rms = (rms_sq / FRAME_LEN as f64).sqrt().max(1e-4) as f32;
            let knee = 0.75 * rms;
            for t in 0..FRAME_LEN {
                let zn = mixed[t] / knee;
                // Hard-ish RF saturation + asymmetric 2nd harmonic distortion
                let distorted = zn.tanh() + 0.30 * (zn.tanh() * zn.tanh()) - 0.12 * zn.abs() * zn.tanh();
                input[t] = distorted * knee;
            }
        }
        ChannelRegime::IndustrialHarsh => {
            // Realistic factory receiver:
            // 1) Nonlinear LNA saturation on signal + colored background,
            // 2) Followed by wideband impulsive EMI transients coupled into ADC/baseband
            //    (unclipped within 12-bit ADC headroom, severely disrupting linear dot products!).
            let (bg_noise, impulses) = split_colored_and_impulses(desired_noise_energy, rng);
            let mut rms_sq = 0.0_f64;
            let mut pre = [0.0_f32; FRAME_LEN];
            for t in 0..FRAME_LEN {
                let z = tx_sig[t] + bg_noise[t];
                pre[t] = z;
                rms_sq += (z as f64) * (z as f64);
            }
            let rms = (rms_sq / FRAME_LEN as f64).sqrt().max(1e-4) as f32;
            let knee = 0.85 * rms;
            for t in 0..FRAME_LEN {
                let zn = pre[t] / knee;
                let sat = (zn.tanh() + 0.25 * zn.tanh() * zn.tanh()) * knee;
                input[t] = sat + impulses[t];
            }
        }
    }

    // AC-coupled ADC removes DC offset
    let mean = input.iter().copied().sum::<f32>() / (FRAME_LEN as f32);
    for x in &mut input {
        *x -= mean;
    }

    FrameSample {
        input,
        target,
        true_lag,
        snr_db: snr_pulse_db,
    }
}

/// Generates a dataset of `count` frames with per-sample pulse SNR in `[snr_min_db, snr_max_db]`.
pub fn generate_dataset(
    regime: ChannelRegime,
    template: &[f32; TEMPLATE_LEN],
    count: usize,
    snr_min_db: f32,
    snr_max_db: f32,
    has_signal: bool,
    rng: &mut Rng,
) -> Vec<FrameSample> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let snr = if (snr_max_db - snr_min_db).abs() < 1e-6 {
            snr_min_db
        } else {
            rng.uniform_f32(snr_min_db, snr_max_db)
        };
        out.push(generate_sample(regime, template, snr, has_signal, rng));
    }
    out
}

fn synthesize_noise(
    regime: ChannelRegime,
    desired_energy: f64,
    rng: &mut Rng,
) -> [f32; FRAME_LEN] {
    let mut raw = [0.0_f64; FRAME_LEN];

    match regime {
        ChannelRegime::Awgn => {
            for slot in &mut raw {
                *slot = rng.normal_f32() as f64;
            }
        }
        ChannelRegime::Colored | ChannelRegime::NonlinearColored => {
            fill_colored_and_tonal(&mut raw, 0.90, 1.35, rng);
        }
        ChannelRegime::IndustrialHarsh => {
            // Handled via split_colored_and_impulses in generate_sample
            fill_colored_and_tonal(&mut raw, 0.88, 1.15, rng);
        }
    }

    scale_to_energy(&raw, desired_energy)
}

/// Splits industrial noise into (colored + narrowband background, heavy-tailed switching impulses)
/// so total energy equals `desired_energy` (45% colored/tonal background + 55% impulsive bursts).
fn split_colored_and_impulses(
    desired_energy: f64,
    rng: &mut Rng,
) -> ([f32; FRAME_LEN], [f32; FRAME_LEN]) {
    let mut bg_raw = [0.0_f64; FRAME_LEN];
    fill_colored_and_tonal(&mut bg_raw, 0.88, 1.25, rng);
    let bg = scale_to_energy(&bg_raw, desired_energy * 0.45);

    // Guarantee at least 3-8 heavy-tailed switching spikes per frame (realistic industrial EMI)
    let mut imp_raw = [0.0_f64; FRAME_LEN];
    let n_spikes = rng.range_usize(3, 9);
    for _ in 0..n_spikes {
        let pos = rng.range_usize(0, FRAME_LEN);
        let sign = if rng.next_f64() < 0.5 { 1.0 } else { -1.0 };
        let mag = 0.6 + 1.4 * rng.next_f64();
        imp_raw[pos] += sign * mag;
        if pos + 1 < FRAME_LEN {
            imp_raw[pos + 1] -= 0.42 * sign * mag;
        }
    }
    let imp = scale_to_energy(&imp_raw, desired_energy * 0.55);
    (bg, imp)
}

fn fill_colored_and_tonal(out: &mut [f64; FRAME_LEN], ar_rho: f64, tone_ratio: f64, rng: &mut Rng) {
    let mut state = 0.0_f64;
    let innov_scale = (1.0 - ar_rho * ar_rho).sqrt();
    for _ in 0..32 {
        state = ar_rho * state + innov_scale * (rng.normal_f32() as f64);
    }

    // Two narrowband industrial carriers (e.g. VFD PWM fundamental + harmonic) overlapping chirp band
    let f_tone1 = 0.09 + 0.23 * rng.next_f64();
    let phi1 = TAU * rng.next_f64();
    let f_tone2 = (f_tone1 + 0.07 + 0.08 * rng.next_f64()).min(0.45);
    let phi2 = TAU * rng.next_f64();

    for (t, slot) in out.iter_mut().enumerate() {
        let white = rng.normal_f32() as f64;
        state = ar_rho * state + innov_scale * white;
        let tone = (TAU * f_tone1 * (t as f64) + phi1).sin()
            + 0.55 * (TAU * f_tone2 * (t as f64) + phi2).sin();
        *slot = 0.25 * white + state + tone_ratio * tone;
    }
}

fn scale_to_energy(raw: &[f64; FRAME_LEN], desired_energy: f64) -> [f32; FRAME_LEN] {
    let current_energy = raw.iter().map(|v| v * v).sum::<f64>().max(1e-12);
    let scale = (desired_energy / current_energy).sqrt();
    let mut out = [0.0_f32; FRAME_LEN];
    for (dst, &src) in out.iter_mut().zip(raw.iter()) {
        *dst = (src * scale) as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chirp_template_is_zero_mean_and_unit_norm() {
        let p = build_lfm_chirp_template();
        let mean: f64 = p.iter().map(|&v| v as f64).sum::<f64>() / TEMPLATE_LEN as f64;
        let energy: f64 = p.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!(mean.abs() < 1e-6, "mean={mean}");
        assert!((energy - 1.0).abs() < 1e-5, "energy={energy}");
    }
}
