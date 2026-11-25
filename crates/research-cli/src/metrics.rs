//! Signal quality metrics (SI-SNR), detection metrics (ROC-AUC, Pd@Pfa, ToA MAE),
//! and ESP32-WROOM-32 latency/energy estimation models.

use neural_filter_core::{FRAME_LEN, ResourceFootprint};

/// Stability floor for SI-SNR computations.
pub const SI_SNR_EPS: f32 = 1e-6;

/// Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB (Luo & Mesgarani, 2019).
///
/// For zero-mean target `s` and estimate `s_hat`:
/// `s_target = (<s_hat, s> / ||s||^2) * s`, `e_noise = s_hat - s_target`,
/// `SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)`.
pub fn si_snr_db(est: &[f32; FRAME_LEN], target: &[f32; FRAME_LEN]) -> f32 {
    let mut dot = 0.0_f64;
    let mut target_energy = 0.0_f64;
    for (&x, &s) in est.iter().zip(target.iter()) {
        let xd = x as f64;
        let sd = s as f64;
        dot += xd * sd;
        target_energy += sd * sd;
    }

    let alpha = dot / (target_energy + 1e-12);
    let mut sig_pow = 0.0_f64;
    let mut err_pow = 0.0_f64;
    for (&x, &s) in est.iter().zip(target.iter()) {
        let t = alpha * (s as f64);
        let e = (x as f64) - t;
        sig_pow += t * t;
        err_pow += e * e;
    }

    (10.0 * ((sig_pow + 1e-12) / (err_pow + 1e-12)).log10()) as f32
}

/// Computes negative SI-SNR loss `-SI-SNR(est, target)` and exact analytical gradient `dL / d(est)`.
///
/// Returns `(loss_value, si_snr_db_value)`.
pub fn neg_si_snr_loss_and_grad(
    est: &[f32; FRAME_LEN],
    target: &[f32; FRAME_LEN],
    grad_out: &mut [f32; FRAME_LEN],
) -> (f32, f32) {
    let eps = SI_SNR_EPS as f64;
    let scale_db = 10.0_f64 / std::f64::consts::LN_10;

    let mut dot = 0.0_f64;
    let mut s_norm_sq = 0.0_f64;
    for (&x, &s) in est.iter().zip(target.iter()) {
        dot += (x as f64) * (s as f64);
        s_norm_sq += (s as f64) * (s as f64);
    }

    let denom_s = s_norm_sq + eps;
    let alpha = dot / denom_s;

    let mut p_t = 0.0_f64;
    let mut p_e = 0.0_f64;
    let mut t_vec = [0.0_f64; FRAME_LEN];
    let mut e_vec = [0.0_f64; FRAME_LEN];
    for i in 0..FRAME_LEN {
        let t = alpha * (target[i] as f64);
        let e = (est[i] as f64) - t;
        t_vec[i] = t;
        e_vec[i] = e;
        p_t += t * t;
        p_e += e * e;
    }

    let si_snr = scale_db * ((p_t + eps) / (p_e + eps)).ln();
    let loss = -si_snr;

    // Backprop:
    //   dL/de = +(2*c / (p_e + eps)) * e
    //   dL/dt = -(2*c / (p_t + eps)) * t - (2*c / (p_e + eps)) * e
    //   dL/d(est) = dL/de + (<dL/dt, s> / denom_s) * s
    let coeff_t = -2.0 * scale_db / (p_t + eps);
    let coeff_e = 2.0 * scale_db / (p_e + eps);

    let mut gt_dot_s = 0.0_f64;
    for i in 0..FRAME_LEN {
        let gt_i = coeff_t * t_vec[i] - coeff_e * e_vec[i];
        gt_dot_s += gt_i * (target[i] as f64);
    }
    let proj_coeff = gt_dot_s / denom_s;

    for i in 0..FRAME_LEN {
        let g = coeff_e * e_vec[i] + proj_coeff * (target[i] as f64);
        grad_out[i] = g as f32;
    }

    (loss as f32, si_snr as f32)
}

/// Detection performance metrics computed from `H1` (signal present) vs `H0` (noise only) scores.
#[derive(Debug, Clone, Copy)]
pub struct DetectionMetrics {
    pub roc_auc: f32,
    pub pd_at_pfa_1pct: f32,
    pub pd_at_pfa_5pct: f32,
}

/// Exact empirical ROC-AUC and `Pd @ Pfa` (1% and 5%) from `H1` and `H0` scalar test statistics.
pub fn compute_detection_metrics(h1_scores: &[f32], h0_scores: &[f32]) -> DetectionMetrics {
    assert!(!h1_scores.is_empty() && !h0_scores.is_empty());

    // Sort H0 descending to determine exact empirical Pfa thresholds.
    let mut h0_sorted = h0_scores.to_vec();
    h0_sorted.sort_by(|a, b| b.total_cmp(a));

    let pd_at_pfa = |pfa: f64| -> f32 {
        let idx = ((pfa * (h0_sorted.len() as f64)).floor() as usize)
            .min(h0_sorted.len().saturating_sub(1));
        let threshold = h0_sorted[idx];
        let hits = h1_scores.iter().filter(|&&s| s > threshold).count();
        (hits as f64 / h1_scores.len() as f64) as f32
    };

    // Exact Mann-Whitney U / ROC-AUC via combined sort
    let mut labeled: Vec<(f32, bool)> = Vec::with_capacity(h1_scores.len() + h0_scores.len());
    labeled.extend(h1_scores.iter().map(|&s| (s, true)));
    labeled.extend(h0_scores.iter().map(|&s| (s, false)));
    labeled.sort_by(|a, b| b.0.total_cmp(&a.0));

    let n_pos = h1_scores.len() as f64;
    let n_neg = h0_scores.len() as f64;
    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut prev_tp = 0.0_f64;
    let mut prev_fp = 0.0_f64;
    let mut auc = 0.0_f64;

    let mut i = 0;
    while i < labeled.len() {
        let score = labeled[i].0;
        while i < labeled.len() && (labeled[i].0 - score).abs() == 0.0 {
            if labeled[i].1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        // Trapezoid area between (prev_fp, prev_tp) and (fp, tp)
        auc += (fp - prev_fp) * 0.5 * (tp + prev_tp);
        prev_fp = fp;
        prev_tp = tp;
    }

    DetectionMetrics {
        roc_auc: (auc / (n_pos * n_neg)) as f32,
        pd_at_pfa_1pct: pd_at_pfa(0.01),
        pd_at_pfa_5pct: pd_at_pfa(0.05),
    }
}

/// ESP32-WROOM-32 analytical cycle, latency, and energy estimation model.
///
/// Datasheet references (Espressif ESP32-WROOM-32, Vdd = 3.3 V, single active core, Wi-Fi/BT off):
/// - 240 MHz active power: ~50 mA * 3.3 V = 165 mW
/// - 160 MHz active power: ~35 mA * 3.3 V = 115.5 mW
/// -  80 MHz active power: ~26 mA * 3.3 V = 85.8 mW
///
/// Cycle-per-MAC factors for Xtensa LX6 (with zero-overhead `loopnez` unit-stride inner loop):
/// - INT8 (`i8 * i8 -> i32`, 2x loads + `mull` + `add` + loop overhead): ~4.5 cycles/MAC
/// - FP32 (`f32 * f32 -> f32`, 2x `lsi` + `madd.s` + loop overhead):     ~5.5 cycles/MAC
///
/// Note: Hardware-exact `CCOUNT` cycle measurements are collected by `firmware/esp32-benchmark`.
#[derive(Debug, Clone, Copy)]
pub struct Esp32EnergyEstimate {
    pub est_cycles: u64,
    pub latency_us_240mhz: f64,
    pub energy_uj_240mhz: f64,
    pub energy_mj_240mhz: f64,
    pub energy_uj_80mhz: f64,
}

impl Esp32EnergyEstimate {
    pub const FREQ_240MHZ: f64 = 240_000_000.0;
    pub const FREQ_80MHZ: f64 = 80_000_000.0;
    pub const POWER_240MHZ_W: f64 = 0.165; // 3.3 V * 50 mA
    pub const POWER_80MHZ_W: f64 = 0.0858; // 3.3 V * 26 mA

    pub fn from_footprint(fp: ResourceFootprint, is_int8: bool) -> Self {
        let cycles_per_mac = if is_int8 { 4.5 } else { 5.5 };
        // Add fixed requantization / activation pass overhead proportional to frame length
        let overhead_cycles = 1_200_u64;
        let est_cycles = (fp.macs_per_frame as f64 * cycles_per_mac).round() as u64 + overhead_cycles;
        let t_240 = est_cycles as f64 / Self::FREQ_240MHZ;
        let t_80 = est_cycles as f64 / Self::FREQ_80MHZ;
        let e_240_j = Self::POWER_240MHZ_W * t_240;
        let e_80_j = Self::POWER_80MHZ_W * t_80;
        Self {
            est_cycles,
            latency_us_240mhz: t_240 * 1e6,
            energy_uj_240mhz: e_240_j * 1e6,
            energy_mj_240mhz: e_240_j * 1e3,
            energy_uj_80mhz: e_80_j * 1e6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn neg_si_snr_analytical_gradient_matches_finite_differences() {
        let mut rng = Rng::from_seed(42);
        let mut est = [0.0_f32; FRAME_LEN];
        let mut target = [0.0_f32; FRAME_LEN];
        for i in 0..FRAME_LEN {
            est[i] = rng.normal_f32() * 0.5;
            target[i] = if (30..62).contains(&i) {
                rng.normal_f32()
            } else {
                0.0
            };
        }

        let mut grad = [0.0_f32; FRAME_LEN];
        let _ = neg_si_snr_loss_and_grad(&est, &target, &mut grad);

        // Compute finite-difference reference in f64 to avoid f32 catastrophic cancellation
        let loss_f64 = |x: &[f64; FRAME_LEN], s: &[f64; FRAME_LEN]| -> f64 {
            let eps = SI_SNR_EPS as f64;
            let dot: f64 = x.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            let s2: f64 = s.iter().map(|b| b * b).sum();
            let alpha = dot / (s2 + eps);
            let mut pt = 0.0;
            let mut pe = 0.0;
            for i in 0..FRAME_LEN {
                let t = alpha * s[i];
                let e = x[i] - t;
                pt += t * t;
                pe += e * e;
            }
            -(10.0 / std::f64::consts::LN_10) * ((pt + eps) / (pe + eps)).ln()
        };

        let est64: [f64; FRAME_LEN] = std::array::from_fn(|i| est[i] as f64);
        let tgt64: [f64; FRAME_LEN] = std::array::from_fn(|i| target[i] as f64);
        let h = 1e-6_f64;
        for i in (0..FRAME_LEN).step_by(3) {
            let mut p = est64;
            let mut m = est64;
            p[i] += h;
            m[i] -= h;
            let num_grad = ((loss_f64(&p, &tgt64) - loss_f64(&m, &tgt64)) / (2.0 * h)) as f32;
            let rel_err = (grad[i] - num_grad).abs() / (num_grad.abs().max(1e-3));
            assert!(
                rel_err < 1e-4,
                "idx={i}: analytical={} numerical={} rel_err={rel_err}",
                grad[i],
                num_grad
            );
        }
    }

    #[test]
    fn roc_auc_is_one_for_perfect_separation() {
        let h1 = [5.0, 6.0, 7.0, 8.0];
        let h0 = [1.0, 2.0, 3.0, 4.0];
        let m = compute_detection_metrics(&h1, &h0);
        assert!((m.roc_auc - 1.0).abs() < 1e-6);
        assert!((m.pd_at_pfa_1pct - 1.0).abs() < 1e-6);
    }
}
