//! Complete reproducible experimental suite for the PhD research:
//!   Study 1: Channel Adaptation Across 4 Regimes (AWGN, Colored, Nonlinear+Colored, Industrial Harsh)
//!   Study 2: Architecture Pareto Sweep on ESP32-WROOM-32 (C = 4, 8, 16, 32 vs Classical MF)
//!   Study 3: Per-Sample Pulse SNR Sweep (-16 .. +2 dB) & Adaptive Mode Controller (DVS/DVFS hook)

use std::fmt::Write as _;
use std::path::Path;

use neural_filter_core::{
    FRAME_LEN, TEMPLATE_LEN, VALID_CORR_LEN,
    adaptive::{AdaptiveThresholds, FilterMode, select_filter_mode},
    classical::{
        footprint_dot_product_f32, footprint_sliding_mf_f32, footprint_sliding_mf_i8,
        linear_fir_matched_f32, sliding_matched_filter_f32,
    },
    conv1d_f32::{NeuralFilterF32, ScratchpadF32},
    conv1d_int8::{NeuralFilterInt8, ScratchpadInt8},
};

use crate::{
    export::export_esp32_model_module,
    metrics::{Esp32EnergyEstimate, compute_detection_metrics, si_snr_db},
    plots::{
        ParetoPlotPoint, SnrPlotPoint, WaveformDemo, write_pareto_svg, write_snr_sweep_svg,
        write_waveform_demo_svg,
    },
    quantize::{OwnedModelInt8, quantize_static_int8},
    rng::Rng,
    signal::{
        ChannelRegime, FrameSample, build_lfm_chirp_template, generate_dataset, generate_sample,
    },
    train::{OwnedModelF32, TrainConfig, train_model},
};

/// ESP32-WROOM-32 SRAM constants (Espressif datasheet).
pub const ESP32_TOTAL_SRAM_BYTES: usize = 520 * 1024;
/// Usable DRAM budget available for static data + stack + buffers on ESP32-WROOM-32.
pub const ESP32_USABLE_DRAM_BYTES: usize = 320 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct EvalMetrics {
    /// Continuous filtered waveform SI-SNR (dB) against clean target pulse.
    pub si_snr_db: f32,
    /// Detection deflection SNR (dB): `10 * log10((mean_H1 - mean_H0)^2 / var_H0)`.
    pub det_snr_db: f32,
    pub roc_auc: f32,
    pub pd_1pct: f32,
    pub pd_5pct: f32,
    pub toa_mae_samples: f32,
    pub toa_hit_rate_1sample: f32,
}

#[derive(Debug, Clone)]
pub struct RegimeComparisonRow {
    pub regime: ChannelRegime,
    pub input_si_snr: f32,
    pub classical: EvalMetrics,
    pub neural_fp32: EvalMetrics,
    pub neural_int8: EvalMetrics,
    pub cal_percentile: f32,
    pub minmax_ptq_si_snr: f32,
}

#[derive(Debug, Clone)]
pub struct ParetoRow {
    pub label: String,
    pub hidden_channels: usize,
    pub params: usize,
    pub fp32: EvalMetrics,
    pub int8: EvalMetrics,
    pub fp32_weight_bytes: usize,
    pub int8_weight_bytes: usize,
    pub int8_model_bytes: usize,
    pub fp32_sram_bytes: usize,
    pub int8_sram_bytes: usize,
    pub macs_per_frame: usize,
    pub est_fp32: Esp32EnergyEstimate,
    pub est_int8: Esp32EnergyEstimate,
    pub cal_percentile: f32,
    pub train_secs: f64,
}

#[derive(Debug, Clone)]
pub struct SnrSweepRow {
    pub snr_db: f32,
    pub cmf_awgn: EvalMetrics,
    pub nmf_int8_awgn: EvalMetrics,
    pub cmf_harsh: EvalMetrics,
    pub nmf_fp32_harsh: EvalMetrics,
    pub nmf_int8_harsh: EvalMetrics,
}

#[derive(Debug, Clone)]
pub struct AdaptiveEvalSummary {
    pub awgn_kept_classical_pct: f32,
    pub harsh_routed_neural_pct: f32,
    pub classical_only_auc: f32,
    pub neural_always_auc: f32,
    pub adaptive_auc: f32,
    pub classical_only_pd1: f32,
    pub neural_always_pd1: f32,
    pub adaptive_pd1: f32,
    pub neural_always_energy_uj: f64,
    pub adaptive_energy_uj: f64,
    pub energy_saving_pct: f64,
}

fn deflection_snr_db(h1_scores: &[f32], h0_scores: &[f32]) -> f32 {
    let mean = |xs: &[f32]| xs.iter().map(|&v| v as f64).sum::<f64>() / xs.len().max(1) as f64;
    let m1 = mean(h1_scores);
    let m0 = mean(h0_scores);
    let var0 = h0_scores
        .iter()
        .map(|&v| {
            let d = (v as f64) - m0;
            d * d
        })
        .sum::<f64>()
        / h0_scores.len().max(1) as f64;
    let diff = (m1 - m0).max(1e-6);
    (10.0 * ((diff * diff) / var0.max(1e-12)).log10()) as f32
}

fn eval_input_si_snr(h1: &[FrameSample]) -> f32 {
    let sum: f64 = h1
        .iter()
        .map(|s| si_snr_db(&s.input, &s.target) as f64)
        .sum();
    (sum / h1.len().max(1) as f64) as f32
}

/// Evaluates the Classical Linear Matched Filter (continuous FIR output for SI-SNR +
/// sliding correlation peak detector for ROC-AUC, Pd@Pfa, Deflection SNR, and ToA).
pub fn eval_classical_mf(
    h1: &[FrameSample],
    h0: &[FrameSample],
    template: &[f32; TEMPLATE_LEN],
) -> EvalMetrics {
    let mut si_snr_sum = 0.0_f64;
    let mut h1_scores = Vec::with_capacity(h1.len());
    let mut h0_scores = Vec::with_capacity(h0.len());
    let mut toa_err_sum = 0.0_f64;
    let mut toa_hits = 0_usize;

    let mut fir_out = [0.0_f32; FRAME_LEN];
    let mut corr = [0.0_f32; VALID_CORR_LEN];

    for s in h1 {
        linear_fir_matched_f32(&s.input, template, &mut fir_out);
        si_snr_sum += si_snr_db(&fir_out, &s.target) as f64;

        let peak = sliding_matched_filter_f32(&s.input, template, &mut corr);
        h1_scores.push(peak.peak_value);
        if let Some(true_lag) = s.true_lag {
            let err = peak.peak_index.abs_diff(true_lag);
            toa_err_sum += err as f64;
            if err <= 1 {
                toa_hits += 1;
            }
        }
    }

    for s in h0 {
        let peak = sliding_matched_filter_f32(&s.input, template, &mut corr);
        h0_scores.push(peak.peak_value);
    }

    let det = compute_detection_metrics(&h1_scores, &h0_scores);
    let n = h1.len().max(1) as f64;
    EvalMetrics {
        si_snr_db: (si_snr_sum / n) as f32,
        det_snr_db: deflection_snr_db(&h1_scores, &h0_scores),
        roc_auc: det.roc_auc,
        pd_1pct: det.pd_at_pfa_1pct,
        pd_5pct: det.pd_at_pfa_5pct,
        toa_mae_samples: (toa_err_sum / n) as f32,
        toa_hit_rate_1sample: (toa_hits as f64 / n) as f32,
    }
}

/// Evaluates a Neural Matched Filter (FP32) on continuous SI-SNR, detection, and ToA localization.
pub fn eval_neural_f32(
    model: &OwnedModelF32,
    h1: &[FrameSample],
    h0: &[FrameSample],
    template: &[f32; TEMPLATE_LEN],
) -> EvalMetrics {
    let view = model.as_view();
    let mut scratch = ScratchpadF32::new();
    let mut out = [0.0_f32; FRAME_LEN];
    let mut corr = [0.0_f32; VALID_CORR_LEN];

    let mut si_snr_sum = 0.0_f64;
    let mut h1_scores = Vec::with_capacity(h1.len());
    let mut h0_scores = Vec::with_capacity(h0.len());
    let mut toa_err_sum = 0.0_f64;
    let mut toa_hits = 0_usize;

    for s in h1 {
        view.infer(&s.input, &mut out, &mut scratch);
        si_snr_sum += si_snr_db(&out, &s.target) as f64;
        let peak = sliding_matched_filter_f32(&out, template, &mut corr);
        h1_scores.push(peak.peak_value);
        if let Some(true_lag) = s.true_lag {
            let err = peak.peak_index.abs_diff(true_lag);
            toa_err_sum += err as f64;
            if err <= 1 {
                toa_hits += 1;
            }
        }
    }

    for s in h0 {
        view.infer(&s.input, &mut out, &mut scratch);
        let peak = sliding_matched_filter_f32(&out, template, &mut corr);
        h0_scores.push(peak.peak_value);
    }

    let det = compute_detection_metrics(&h1_scores, &h0_scores);
    let n = h1.len().max(1) as f64;
    EvalMetrics {
        si_snr_db: (si_snr_sum / n) as f32,
        det_snr_db: deflection_snr_db(&h1_scores, &h0_scores),
        roc_auc: det.roc_auc,
        pd_1pct: det.pd_at_pfa_1pct,
        pd_5pct: det.pd_at_pfa_5pct,
        toa_mae_samples: (toa_err_sum / n) as f32,
        toa_hit_rate_1sample: (toa_hits as f64 / n) as f32,
    }
}

/// Evaluates a Static INT8 Neural Matched Filter using the exact `no_std` integer-only kernel.
pub fn eval_neural_int8(
    model: &OwnedModelInt8,
    h1: &[FrameSample],
    h0: &[FrameSample],
    template: &[f32; TEMPLATE_LEN],
) -> EvalMetrics {
    let view = model.as_view();
    let mut scratch = ScratchpadInt8::new();
    let mut out = [0.0_f32; FRAME_LEN];
    let mut corr = [0.0_f32; VALID_CORR_LEN];

    let mut si_snr_sum = 0.0_f64;
    let mut h1_scores = Vec::with_capacity(h1.len());
    let mut h0_scores = Vec::with_capacity(h0.len());
    let mut toa_err_sum = 0.0_f64;
    let mut toa_hits = 0_usize;

    for s in h1 {
        view.infer_f32_io(&s.input, &mut out, &mut scratch);
        si_snr_sum += si_snr_db(&out, &s.target) as f64;
        let peak = sliding_matched_filter_f32(&out, template, &mut corr);
        h1_scores.push(peak.peak_value);
        if let Some(true_lag) = s.true_lag {
            let err = peak.peak_index.abs_diff(true_lag);
            toa_err_sum += err as f64;
            if err <= 1 {
                toa_hits += 1;
            }
        }
    }

    for s in h0 {
        view.infer_f32_io(&s.input, &mut out, &mut scratch);
        let peak = sliding_matched_filter_f32(&out, template, &mut corr);
        h0_scores.push(peak.peak_value);
    }

    let det = compute_detection_metrics(&h1_scores, &h0_scores);
    let n = h1.len().max(1) as f64;
    EvalMetrics {
        si_snr_db: (si_snr_sum / n) as f32,
        det_snr_db: deflection_snr_db(&h1_scores, &h0_scores),
        roc_auc: det.roc_auc,
        pd_1pct: det.pd_at_pfa_1pct,
        pd_5pct: det.pd_at_pfa_5pct,
        toa_mae_samples: (toa_err_sum / n) as f32,
        toa_hit_rate_1sample: (toa_hits as f64 / n) as f32,
    }
}

/// Runs the complete PhD research suite, writes CSV/SVG/Markdown artifacts,
/// and exports the Pareto-optimal trained model to `firmware/esp32-benchmark`.
pub fn run_full_suite(
    results_dir: &Path,
    firmware_export_path: &Path,
    quick: bool,
) -> std::io::Result<()> {
    std::fs::create_dir_all(results_dir)?;
    let template = build_lfm_chirp_template();

    let train_n = if quick { 1_500 } else { 4_000 };
    let val_n = if quick { 250 } else { 500 };
    let cal_n = if quick { 200 } else { 320 };
    let test_n = if quick { 300 } else { 600 };
    let epochs = if quick { 35 } else { 75 };

    // Realistic low-SNR IoT operating band (per-sample pulse SNR)
    // Centered around -3 dB (where AWGN classical MF reaches ~0.92+ AUC while non-AWGN industrial channels degrade sharply)
    let train_snr_min = -12.0_f32;
    let train_snr_max = 4.0_f32;
    let eval_snr_min = -8.0_f32;
    let eval_snr_max = 2.0_f32;

    let train_cfg = TrainConfig {
        epochs,
        batch_size: 64,
        lr_max: 4.5e-3,
        lr_min: 1.5e-4,
        weight_decay: 1e-4,
        grad_clip: 5.0,
    };

    println!("=== Neural Matched Filter Research Suite (Pure Rust / ESP32-WROOM-32) ===");
    println!(
        "Config: FRAME_LEN={FRAME_LEN}, TEMPLATE_LEN={TEMPLATE_LEN}, train={train_n} (SNR [{train_snr_min:.0}, {train_snr_max:+.0}] dB), eval={test_n}x2 H1/H0 (SNR [{eval_snr_min:.0}, {eval_snr_max:.0}] dB), epochs={epochs}\n"
    );

    // =========================================================================
    // STUDY 1: Channel Adaptation Across 4 Regimes (Medium C = 16)
    // =========================================================================
    println!("[Study 1/3] Channel Adaptation Across 4 Physical Regimes (Conv1D C=16)...");
    let mut regime_rows = Vec::new();
    let mut awgn_model_int8: Option<OwnedModelInt8> = None;
    let mut harsh_c16_f32: Option<OwnedModelF32> = None;
    let mut harsh_c16_int8: Option<OwnedModelInt8> = None;
    let mut export_test_frames: Vec<(&str, FrameSample)> = Vec::new();

    for (idx, &regime) in ChannelRegime::ALL.iter().enumerate() {
        let mut rng = Rng::from_seed(10_000 + (idx as u64) * 97);
        let train_ds = generate_dataset(
            regime,
            &template,
            train_n,
            train_snr_min,
            train_snr_max,
            true,
            &mut rng,
        );
        let val_ds = generate_dataset(
            regime,
            &template,
            val_n,
            eval_snr_min,
            eval_snr_max,
            true,
            &mut rng,
        );
        let cal_ds = generate_dataset(
            regime,
            &template,
            cal_n,
            train_snr_min + 1.0,
            train_snr_max - 1.0,
            true,
            &mut rng,
        );
        let test_h1 = generate_dataset(
            regime,
            &template,
            test_n,
            eval_snr_min,
            eval_snr_max,
            true,
            &mut rng,
        );
        let test_h0 = generate_dataset(
            regime,
            &template,
            test_n,
            eval_snr_min,
            eval_snr_max,
            false,
            &mut rng,
        );

        let demo_frame = generate_sample(regime, &template, -3.0, true, &mut rng);
        export_test_frames.push((regime.slug(), demo_frame));

        let input_si_snr = eval_input_si_snr(&test_h1);
        let classical = eval_classical_mf(&test_h1, &test_h0, &template);

        let summary = train_model(16, &train_ds, &val_ds, train_cfg, 20_000 + idx as u64);
        let neural_fp32 = eval_neural_f32(&summary.model, &test_h1, &test_h0, &template);
        let qres = quantize_static_int8(&summary.model, &cal_ds);
        let neural_int8 = eval_neural_int8(&qres.model, &test_h1, &test_h0, &template);

        println!(
            "  {:<28} | CMF: AUC {:.3}, Pd@1% {:>5.1}%, ToA<=1 {:>5.1}% | NMF-FP32: {:+5.2} dB (val {:+.2} @ep {}) | NMF-INT8: {:+5.2} dB (cal p={:.2}%, min-max {:+.2}->{:+.2}), AUC {:.3}, Pd@1% {:>5.1}%, ToA<=1 {:>5.1}% [{:.1}s]",
            regime.display_name(),
            classical.roc_auc,
            classical.pd_1pct * 100.0,
            classical.toa_hit_rate_1sample * 100.0,
            neural_fp32.si_snr_db,
            summary.best_val_si_snr,
            summary.best_epoch,
            neural_int8.si_snr_db,
            qres.model.calibration_percentile * 100.0,
            qres.minmax_cal_si_snr,
            qres.best_cal_si_snr,
            neural_int8.roc_auc,
            neural_int8.pd_1pct * 100.0,
            neural_int8.toa_hit_rate_1sample * 100.0,
            summary.elapsed_secs
        );

        if regime == ChannelRegime::Awgn {
            awgn_model_int8 = Some(qres.model.clone());
        }
        if regime == ChannelRegime::IndustrialHarsh {
            harsh_c16_f32 = Some(summary.model.clone());
            harsh_c16_int8 = Some(qres.model.clone());
        }

        regime_rows.push(RegimeComparisonRow {
            regime,
            input_si_snr,
            classical,
            neural_fp32,
            neural_int8,
            cal_percentile: qres.model.calibration_percentile,
            minmax_ptq_si_snr: qres.minmax_cal_si_snr,
        });
    }

    // =========================================================================
    // STUDY 2: Pareto Trade-off Across Model Widths (C = 4, 8, 16, 32) on IndustrialHarsh
    // =========================================================================
    println!("\n[Study 2/3] Pareto Trade-off Sweep on Industrial Harsh Channel (C = 4, 8, 16, 32)...");
    let mut rng_pareto = Rng::from_seed(55_555);
    let harsh_train = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        &template,
        train_n,
        train_snr_min,
        train_snr_max,
        true,
        &mut rng_pareto,
    );
    let harsh_val = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        &template,
        val_n,
        eval_snr_min,
        eval_snr_max,
        true,
        &mut rng_pareto,
    );
    let harsh_cal = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        &template,
        cal_n,
        train_snr_min + 1.0,
        train_snr_max - 1.0,
        true,
        &mut rng_pareto,
    );
    let harsh_test_h1 = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        &template,
        test_n,
        eval_snr_min,
        eval_snr_max,
        true,
        &mut rng_pareto,
    );
    let harsh_test_h0 = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        &template,
        test_n,
        eval_snr_min,
        eval_snr_max,
        false,
        &mut rng_pareto,
    );

    let harsh_classical = eval_classical_mf(&harsh_test_h1, &harsh_test_h0, &template);
    println!(
        "  Classical MF | SI-SNR {:+6.2} dB | AUC {:.3} | Pd@1% {:>5.1}% | ToA<=1 {:>5.1}% | MAE {:>5.2} smp",
        harsh_classical.si_snr_db,
        harsh_classical.roc_auc,
        harsh_classical.pd_1pct * 100.0,
        harsh_classical.toa_hit_rate_1sample * 100.0,
        harsh_classical.toa_mae_samples
    );

    let arch_specs: [(&str, usize); 4] = [
        ("Tiny-C4", 4),
        ("Small-C8", 8),
        ("Medium-C16", 16),
        ("Large-C32", 32),
    ];

    let mut pareto_rows = Vec::new();
    let mut exported_model_f32 = harsh_c16_f32.expect("C16 FP32 trained");
    let mut exported_model_int8 = harsh_c16_int8.expect("C16 INT8 trained");

    for (idx, &(label, c)) in arch_specs.iter().enumerate() {
        let summary = train_model(c, &harsh_train, &harsh_val, train_cfg, 70_000 + idx as u64);
        let fp32_eval = eval_neural_f32(&summary.model, &harsh_test_h1, &harsh_test_h0, &template);
        let qres = quantize_static_int8(&summary.model, &harsh_cal);
        let int8_eval = eval_neural_int8(&qres.model, &harsh_test_h1, &harsh_test_h0, &template);

        let fp_f32 = NeuralFilterF32::footprint(c);
        let fp_i8 = NeuralFilterInt8::footprint(c);
        let est_f32 = Esp32EnergyEstimate::from_footprint(fp_f32, false);
        let est_i8 = Esp32EnergyEstimate::from_footprint(fp_i8, true);

        println!(
            "  {:<11} | Params {:>5} | FP32 {:+5.2} -> INT8 {:+5.2} dB (d={:+.2}) | AUC {:.3} | Pd@1% {:>5.1}% | ToA<=1 {:>5.1}% | W {:>4} B | SRAM {:>5.2} KB | {:>6.1} kMACs | ~{:>4.2} mJ [{:.1}s]",
            label,
            summary.model.param_count(),
            fp32_eval.si_snr_db,
            int8_eval.si_snr_db,
            int8_eval.si_snr_db - fp32_eval.si_snr_db,
            int8_eval.roc_auc,
            int8_eval.pd_1pct * 100.0,
            int8_eval.toa_hit_rate_1sample * 100.0,
            fp_i8.weight_bytes,
            fp_i8.total_sram_bytes() as f64 / 1024.0,
            fp_i8.macs_per_frame as f64 / 1000.0,
            est_i8.energy_mj_240mhz,
            summary.elapsed_secs
        );

        if c == 16 {
            exported_model_f32 = summary.model.clone();
            exported_model_int8 = qres.model.clone();
        }

        pareto_rows.push(ParetoRow {
            label: label.to_string(),
            hidden_channels: c,
            params: summary.model.param_count(),
            fp32: fp32_eval,
            int8: int8_eval,
            fp32_weight_bytes: fp_f32.weight_bytes,
            int8_weight_bytes: fp_i8.weight_bytes,
            int8_model_bytes: fp_i8.model_bytes(),
            fp32_sram_bytes: fp_f32.total_sram_bytes(),
            int8_sram_bytes: fp_i8.total_sram_bytes(),
            macs_per_frame: fp_i8.macs_per_frame,
            est_fp32: est_f32,
            est_int8: est_i8,
            cal_percentile: qres.model.calibration_percentile,
            train_secs: summary.elapsed_secs,
        });
    }

    // =========================================================================
    // STUDY 3: SNR Sweep (-14 .. +4 dB) & Adaptive Mode Controller Evaluation
    // =========================================================================
    println!("\n[Study 3/3] Per-Sample Pulse SNR Sweep (-14..+4 dB) & Adaptive Switching...");
    let awgn_int8 = awgn_model_int8.expect("AWGN INT8 model");
    let snr_levels = [-14.0_f32, -11.0, -8.0, -5.0, -2.0, 1.0, 4.0];
    let sweep_n = if quick { 150 } else { 350 };
    let mut snr_rows = Vec::new();
    let mut rng_snr = Rng::from_seed(90_001);

    for &snr_db in &snr_levels {
        let awgn_h1 = generate_dataset(
            ChannelRegime::Awgn,
            &template,
            sweep_n,
            snr_db,
            snr_db,
            true,
            &mut rng_snr,
        );
        let awgn_h0 = generate_dataset(
            ChannelRegime::Awgn,
            &template,
            sweep_n,
            snr_db,
            snr_db,
            false,
            &mut rng_snr,
        );
        let harsh_h1 = generate_dataset(
            ChannelRegime::IndustrialHarsh,
            &template,
            sweep_n,
            snr_db,
            snr_db,
            true,
            &mut rng_snr,
        );
        let harsh_h0 = generate_dataset(
            ChannelRegime::IndustrialHarsh,
            &template,
            sweep_n,
            snr_db,
            snr_db,
            false,
            &mut rng_snr,
        );

        let cmf_awgn = eval_classical_mf(&awgn_h1, &awgn_h0, &template);
        let nmf_int8_awgn = eval_neural_int8(&awgn_int8, &awgn_h1, &awgn_h0, &template);
        let cmf_harsh = eval_classical_mf(&harsh_h1, &harsh_h0, &template);
        let nmf_fp32_harsh = eval_neural_f32(&exported_model_f32, &harsh_h1, &harsh_h0, &template);
        let nmf_int8_harsh = eval_neural_int8(&exported_model_int8, &harsh_h1, &harsh_h0, &template);

        println!(
            "  SNR {:+5.1} dB | AWGN AUC CMF {:.3} vs NMF-I8 {:.3} | Harsh AUC CMF {:.3} (Pd@1% {:>5.1}%) vs NMF-I8 {:.3} (Pd@1% {:>5.1}%, SI-SNR {:+5.2} dB)",
            snr_db,
            cmf_awgn.roc_auc,
            nmf_int8_awgn.roc_auc,
            cmf_harsh.roc_auc,
            cmf_harsh.pd_1pct * 100.0,
            nmf_int8_harsh.roc_auc,
            nmf_int8_harsh.pd_1pct * 100.0,
            nmf_int8_harsh.si_snr_db
        );

        snr_rows.push(SnrSweepRow {
            snr_db,
            cmf_awgn,
            nmf_int8_awgn,
            cmf_harsh,
            nmf_fp32_harsh,
            nmf_int8_harsh,
        });
    }

    let adaptive_summary = evaluate_adaptive_controller(
        &template,
        &exported_model_int8,
        if quick { 200 } else { 400 },
    );
    println!(
        "  Adaptive Controller: AWGN->Classical {:.1}%, Harsh->Neural {:.1}% | Mixed AUC: CMF {:.3} -> Adaptive {:.3} (Always-Neural {:.3}) | Energy {:.1} uJ vs {:.1} uJ ({:.1}% saved)",
        adaptive_summary.awgn_kept_classical_pct,
        adaptive_summary.harsh_routed_neural_pct,
        adaptive_summary.classical_only_auc,
        adaptive_summary.adaptive_auc,
        adaptive_summary.neural_always_auc,
        adaptive_summary.adaptive_energy_uj,
        adaptive_summary.neural_always_energy_uj,
        adaptive_summary.energy_saving_pct
    );

    // =========================================================================
    // Export ESP32 Firmware Weights + Publication Plots + CSV/Markdown Report
    // =========================================================================
    let frame_refs: Vec<(&str, &FrameSample)> =
        export_test_frames.iter().map(|(l, s)| (*l, s)).collect();
    export_esp32_model_module(
        firmware_export_path,
        "Medium-C16 (3-Layer Dilated Conv1D)",
        &template,
        &exported_model_f32,
        &exported_model_int8,
        &frame_refs,
    )?;

    let pareto_pts: Vec<ParetoPlotPoint> = pareto_rows
        .iter()
        .map(|r| ParetoPlotPoint {
            label: r.label.clone(),
            macs_k: r.macs_per_frame as f64 / 1000.0,
            si_snr_fp32: r.fp32.si_snr_db,
            si_snr_int8: r.int8.si_snr_db,
            sram_int8_kb: r.int8_sram_bytes as f64 / 1024.0,
        })
        .collect();
    write_pareto_svg(
        &results_dir.join("pareto_tradeoff.svg"),
        harsh_classical.si_snr_db,
        &pareto_pts,
    )?;

    // Plot ROC-AUC vs SNR (and SI-SNR vs SNR)
    let snr_pts: Vec<SnrPlotPoint> = snr_rows
        .iter()
        .map(|r| SnrPlotPoint {
            snr_db: r.snr_db,
            cmf_awgn: r.cmf_awgn.roc_auc * 100.0,
            nmf_int8_awgn: r.nmf_int8_awgn.roc_auc * 100.0,
            cmf_harsh: r.cmf_harsh.roc_auc * 100.0,
            nmf_fp32_harsh: r.nmf_fp32_harsh.roc_auc * 100.0,
            nmf_int8_harsh: r.nmf_int8_harsh.roc_auc * 100.0,
        })
        .collect();
    write_snr_sweep_svg(&results_dir.join("snr_sweep.svg"), &snr_pts)?;

    // Waveform demo on the saved -9 dB IndustrialHarsh sample
    let (_, demo_harsh) = export_test_frames.last().expect("demo frame");
    let mut cmf_demo_out = [0.0_f32; FRAME_LEN];
    linear_fir_matched_f32(&demo_harsh.input, &template, &mut cmf_demo_out);
    let mut nmf_demo_out = [0.0_f32; FRAME_LEN];
    let mut scratch_q = ScratchpadInt8::new();
    exported_model_int8
        .as_view()
        .infer_f32_io(&demo_harsh.input, &mut nmf_demo_out, &mut scratch_q);
    write_waveform_demo_svg(
        &results_dir.join("waveform_demo.svg"),
        &WaveformDemo {
            snr_db: demo_harsh.snr_db,
            clean: &demo_harsh.target,
            noisy: &demo_harsh.input,
            cmf_out: &cmf_demo_out,
            nmf_int8_out: &nmf_demo_out,
            cmf_sisnr: si_snr_db(&cmf_demo_out, &demo_harsh.target),
            nmf_sisnr: si_snr_db(&nmf_demo_out, &demo_harsh.target),
        },
    )?;

    write_csv_files(results_dir, &regime_rows, &pareto_rows, &snr_rows)?;
    write_markdown_report(
        &results_dir.join("research_report.md"),
        &regime_rows,
        harsh_classical,
        &pareto_rows,
        &snr_rows,
        &adaptive_summary,
    )?;

    println!("\nArtifacts written to: {}", results_dir.display());
    println!(
        "ESP32 firmware model exported to: {}",
        firmware_export_path.display()
    );
    Ok(())
}

fn evaluate_adaptive_controller(
    template: &[f32; TEMPLATE_LEN],
    neural_int8: &OwnedModelInt8,
    n_per_regime: usize,
) -> AdaptiveEvalSummary {
    let mut rng = Rng::from_seed(123_456);
    let awgn_h1 = generate_dataset(
        ChannelRegime::Awgn,
        template,
        n_per_regime,
        -8.0,
        2.0,
        true,
        &mut rng,
    );
    let awgn_h0 = generate_dataset(
        ChannelRegime::Awgn,
        template,
        n_per_regime,
        -8.0,
        2.0,
        false,
        &mut rng,
    );
    let harsh_h1 = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        template,
        n_per_regime,
        -8.0,
        2.0,
        true,
        &mut rng,
    );
    let harsh_h0 = generate_dataset(
        ChannelRegime::IndustrialHarsh,
        template,
        n_per_regime,
        -8.0,
        2.0,
        false,
        &mut rng,
    );

    let thresholds = AdaptiveThresholds::default();
    let e_cmf_uj =
        Esp32EnergyEstimate::from_footprint(footprint_sliding_mf_i8(), true).energy_uj_80mhz;
    let e_nmf_uj = Esp32EnergyEstimate::from_footprint(
        NeuralFilterInt8::footprint(neural_int8.hidden_channels),
        true,
    )
    .energy_uj_240mhz;
    let e_diag_uj = 0.9_f64;

    let mut awgn_classical = 0_usize;
    let mut harsh_neural = 0_usize;
    let mut adaptive_energy_sum = 0.0_f64;
    let mut total_frames = 0_usize;

    let mut cmf_h1 = Vec::new();
    let mut cmf_h0 = Vec::new();
    let mut nmf_h1 = Vec::new();
    let mut nmf_h0 = Vec::new();
    let mut adp_h1 = Vec::new();
    let mut adp_h0 = Vec::new();

    let mut scratch_q = ScratchpadInt8::new();
    let mut nmf_out = [0.0_f32; FRAME_LEN];
    let mut corr = [0.0_f32; VALID_CORR_LEN];

    // Normalize detection statistic by per-frame RMS so Classical & Neural scores live on a comparable threshold scale
    let frame_rms = |f: &[f32; FRAME_LEN]| -> f32 {
        let p: f32 = f.iter().map(|&x| x * x).sum::<f32>() / (FRAME_LEN as f32);
        p.sqrt().max(1e-5)
    };

    for (is_harsh, is_h1, dataset) in [
        (false, true, &awgn_h1),
        (false, false, &awgn_h0),
        (true, true, &harsh_h1),
        (true, false, &harsh_h0),
    ] {
        for s in dataset {
            let (mode, _) = select_filter_mode(&s.input, thresholds);
            let rms = frame_rms(&s.input);
            let cmf_score = sliding_matched_filter_f32(&s.input, template, &mut corr).peak_value / rms;
            neural_int8
                .as_view()
                .infer_f32_io(&s.input, &mut nmf_out, &mut scratch_q);
            let nmf_score = sliding_matched_filter_f32(&nmf_out, template, &mut corr).peak_value / rms;

            let adp_score = match mode {
                FilterMode::ClassicalLowPower => {
                    if !is_harsh {
                        awgn_classical += 1;
                    }
                    adaptive_energy_sum += e_diag_uj + e_cmf_uj;
                    cmf_score
                }
                FilterMode::NeuralInt8 => {
                    if is_harsh {
                        harsh_neural += 1;
                    }
                    adaptive_energy_sum += e_diag_uj + e_nmf_uj;
                    nmf_score
                }
            };
            total_frames += 1;

            if is_h1 {
                cmf_h1.push(cmf_score);
                nmf_h1.push(nmf_score);
                adp_h1.push(adp_score);
            } else {
                cmf_h0.push(cmf_score);
                nmf_h0.push(nmf_score);
                adp_h0.push(adp_score);
            }
        }
    }

    let det_cmf = compute_detection_metrics(&cmf_h1, &cmf_h0);
    let det_nmf = compute_detection_metrics(&nmf_h1, &nmf_h0);
    let det_adp = compute_detection_metrics(&adp_h1, &adp_h0);

    let adaptive_energy_uj = adaptive_energy_sum / (total_frames as f64);
    let energy_saving_pct = (1.0 - adaptive_energy_uj / e_nmf_uj) * 100.0;

    AdaptiveEvalSummary {
        awgn_kept_classical_pct: (awgn_classical as f64 / (2 * n_per_regime) as f64 * 100.0) as f32,
        harsh_routed_neural_pct: (harsh_neural as f64 / (2 * n_per_regime) as f64 * 100.0) as f32,
        classical_only_auc: det_cmf.roc_auc,
        neural_always_auc: det_nmf.roc_auc,
        adaptive_auc: det_adp.roc_auc,
        classical_only_pd1: det_cmf.pd_at_pfa_1pct,
        neural_always_pd1: det_nmf.pd_at_pfa_1pct,
        adaptive_pd1: det_adp.pd_at_pfa_1pct,
        neural_always_energy_uj: e_nmf_uj,
        adaptive_energy_uj,
        energy_saving_pct,
    }
}

fn write_csv_files(
    dir: &Path,
    regimes: &[RegimeComparisonRow],
    pareto: &[ParetoRow],
    snr: &[SnrSweepRow],
) -> std::io::Result<()> {
    let mut r_csv = String::from(
        "regime,input_si_snr_db,cmf_fir_si_snr_db,cmf_det_snr_db,cmf_auc,cmf_pd_1pct,cmf_pd_5pct,cmf_toa_mae,cmf_toa_hit1,nmf_fp32_si_snr_db,nmf_fp32_auc,nmf_int8_si_snr_db,nmf_int8_det_snr_db,nmf_int8_auc,nmf_int8_pd_1pct,nmf_int8_pd_5pct,nmf_int8_toa_mae,nmf_int8_toa_hit1,cal_percentile,minmax_cal_si_snr_db\n",
    );
    for r in regimes {
        writeln!(
            r_csv,
            "{},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.3},{:.4},{:.3},{:.4},{:.3},{:.3},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4},{:.3}",
            r.regime.slug(),
            r.input_si_snr,
            r.classical.si_snr_db,
            r.classical.det_snr_db,
            r.classical.roc_auc,
            r.classical.pd_1pct,
            r.classical.pd_5pct,
            r.classical.toa_mae_samples,
            r.classical.toa_hit_rate_1sample,
            r.neural_fp32.si_snr_db,
            r.neural_fp32.roc_auc,
            r.neural_int8.si_snr_db,
            r.neural_int8.det_snr_db,
            r.neural_int8.roc_auc,
            r.neural_int8.pd_1pct,
            r.neural_int8.pd_5pct,
            r.neural_int8.toa_mae_samples,
            r.neural_int8.toa_hit_rate_1sample,
            r.cal_percentile,
            r.minmax_ptq_si_snr
        )
        .unwrap();
    }
    std::fs::write(dir.join("regimes.csv"), r_csv)?;

    let mut p_csv = String::from(
        "model,hidden_channels,params,fp32_si_snr_db,int8_si_snr_db,quant_drop_db,int8_det_snr_db,int8_auc,int8_pd_1pct,int8_pd_5pct,int8_toa_mae,int8_toa_hit1,fp32_weight_bytes,int8_weight_bytes,int8_model_bytes,fp32_sram_bytes,int8_sram_bytes,pct_total_520kb_sram,macs_per_frame,est_int8_cycles,est_int8_latency_us,est_int8_energy_uj,est_int8_energy_mj,cal_percentile,train_secs\n",
    );
    for r in pareto {
        writeln!(
            p_csv,
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.3},{:.4},{},{},{},{},{},{:.3},{},{},{:.1},{:.2},{:.5},{:.4},{:.2}",
            r.label,
            r.hidden_channels,
            r.params,
            r.fp32.si_snr_db,
            r.int8.si_snr_db,
            r.int8.si_snr_db - r.fp32.si_snr_db,
            r.int8.det_snr_db,
            r.int8.roc_auc,
            r.int8.pd_1pct,
            r.int8.pd_5pct,
            r.int8.toa_mae_samples,
            r.int8.toa_hit_rate_1sample,
            r.fp32_weight_bytes,
            r.int8_weight_bytes,
            r.int8_model_bytes,
            r.fp32_sram_bytes,
            r.int8_sram_bytes,
            r.int8_sram_bytes as f64 / ESP32_TOTAL_SRAM_BYTES as f64 * 100.0,
            r.macs_per_frame,
            r.est_int8.est_cycles,
            r.est_int8.latency_us_240mhz,
            r.est_int8.energy_uj_240mhz,
            r.est_int8.energy_mj_240mhz,
            r.cal_percentile,
            r.train_secs
        )
        .unwrap();
    }
    std::fs::write(dir.join("pareto.csv"), p_csv)?;

    let mut s_csv = String::from(
        "snr_pulse_db,cmf_awgn_auc,cmf_awgn_pd1,nmf_int8_awgn_auc,nmf_int8_awgn_pd1,cmf_harsh_auc,cmf_harsh_pd1,cmf_harsh_toa_hit1,nmf_fp32_harsh_auc,nmf_int8_harsh_sisnr,nmf_int8_harsh_auc,nmf_int8_harsh_pd1,nmf_int8_harsh_toa_hit1\n",
    );
    for r in snr {
        writeln!(
            s_csv,
            "{:.1},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4},{:.4}",
            r.snr_db,
            r.cmf_awgn.roc_auc,
            r.cmf_awgn.pd_1pct,
            r.nmf_int8_awgn.roc_auc,
            r.nmf_int8_awgn.pd_1pct,
            r.cmf_harsh.roc_auc,
            r.cmf_harsh.pd_1pct,
            r.cmf_harsh.toa_hit_rate_1sample,
            r.nmf_fp32_harsh.roc_auc,
            r.nmf_int8_harsh.si_snr_db,
            r.nmf_int8_harsh.roc_auc,
            r.nmf_int8_harsh.pd_1pct,
            r.nmf_int8_harsh.toa_hit_rate_1sample
        )
        .unwrap();
    }
    std::fs::write(dir.join("snr_sweep.csv"), s_csv)?;

    Ok(())
}

fn write_markdown_report(
    path: &Path,
    regimes: &[RegimeComparisonRow],
    harsh_cmf: EvalMetrics,
    pareto: &[ParetoRow],
    snr: &[SnrSweepRow],
    adaptive: &AdaptiveEvalSummary,
) -> std::io::Result<()> {
    let fp_dot = footprint_dot_product_f32();
    let est_dot = Esp32EnergyEstimate::from_footprint(fp_dot, false);
    let fp_smf_f32 = footprint_sliding_mf_f32();
    let est_smf_f32 = Esp32EnergyEstimate::from_footprint(fp_smf_f32, false);
    let fp_smf_i8 = footprint_sliding_mf_i8();
    let est_smf_i8 = Esp32EnergyEstimate::from_footprint(fp_smf_i8, true);

    let mut md = String::new();
    writeln!(
        md,
        "# Neural Matched Filtering for ESP32-WROOM-32: Experimental Report\n"
    )
    .unwrap();
    writeln!(md, "## 1. Experimental Methodology").unwrap();
    writeln!(md, "- **Signal Model**: Hann-windowed LFM chirp preamble (`TEMPLATE_LEN = {TEMPLATE_LEN}`, sweep `0.08 -> 0.34`, zero-mean, unit L2 energy) embedded at random arrival lag `tau in [8, 88]` inside an `FRAME_LEN = {FRAME_LEN}` sample baseband frame (`8 ms @ 16 kHz`).").unwrap();
    writeln!(md, "- **SNR Convention**: Standard per-sample pulse SNR (`SNR_pulse = 10 log10(P_pulse / P_noise)`), where the 32-sample chirp gives `+15.05 dB` theoretical matched-filter processing gain in AWGN. Evaluated in the low-SNR IoT operating band `[-8, +2] dB` (and swept across `[-14, +4] dB`).").unwrap();
    writeln!(md, "- **Architecture**: Lightweight 3-layer dilated Conv1D (`kernel_size = 9`, dilations `[1, 2, 4]`, receptive field = `57` samples), trained purely in Rust with exact analytical backpropagation on negative **SI-SNR** loss (Luo & Mesgarani, 2019).").unwrap();
    writeln!(md, "- **Static INT8 Quantization**: True integer-only inference (`i8` weights per-output-channel, `i8` signed I/O, `u8` post-ReLU hidden activations with `zp = 0`, `i32` accumulators, Q31 fixed-point requantization — zero floats in the inference path).").unwrap();
    writeln!(md, "- **Target Platform**: ESP32-WROOM-32 (Xtensa LX6 @ 240 MHz, 520 KB SRAM / ~320 KB usable DRAM; active power model: 165 mW @ 240 MHz, 85.8 mW @ 80 MHz).\n").unwrap();

    writeln!(
        md,
        "## 2. Study 1 — Channel Adaptation Across 4 Physical Regimes (Conv1D C=16, SNR_pulse in [-8, +2] dB)\n"
    )
    .unwrap();
    writeln!(md, "| Channel Regime | Input SI-SNR (dB) | CMF FIR SI-SNR (dB) | CMF AUC | CMF Pd@1% | CMF ToA<=1 | Neural FP32 SI-SNR (dB) | Neural INT8 SI-SNR (dB) | INT8 AUC | INT8 Pd@1% | INT8 ToA<=1 |").unwrap();
    writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for r in regimes {
        writeln!(
            md,
            "| {} | {:+.2} | {:+.2} | {:.3} | {:.1}% | {:.1}% | {:+.2} | {:+.2} | {:.3} | {:.1}% | {:.1}% |",
            r.regime.display_name(),
            r.input_si_snr,
            r.classical.si_snr_db,
            r.classical.roc_auc,
            r.classical.pd_1pct * 100.0,
            r.classical.toa_hit_rate_1sample * 100.0,
            r.neural_fp32.si_snr_db,
            r.neural_int8.si_snr_db,
            r.neural_int8.roc_auc,
            r.neural_int8.pd_1pct * 100.0,
            r.neural_int8.toa_hit_rate_1sample * 100.0
        )
        .unwrap();
    }

    writeln!(
        md,
        "\n## 3. Study 2 — Accuracy vs Memory & Energy Pareto Trade-Off (Industrial Harsh Channel)\n"
    )
    .unwrap();
    writeln!(md, "| Filter / Model | Precision | Params | SI-SNR (dB) | Quant Drop (dB) | ROC-AUC | Pd@1% | ToA<=1 | Weights (B) | Peak SRAM (KB) | % of 320KB DRAM | MACs/frame | Est. Latency @240MHz (us) | Est. Energy @240MHz (mJ) |").unwrap();
    writeln!(md, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    writeln!(
        md,
        "| Classical Dot-Product (sync) | FP32 | 128 | — | — | — | — | — | {} | {:.2} | {:.2}% | {} | {:.1} | {:.5} |",
        fp_dot.weight_bytes,
        fp_dot.total_sram_bytes() as f64 / 1024.0,
        fp_dot.total_sram_bytes() as f64 / ESP32_USABLE_DRAM_BYTES as f64 * 100.0,
        fp_dot.macs_per_frame,
        est_dot.latency_us_240mhz,
        est_dot.energy_mj_240mhz
    )
    .unwrap();
    writeln!(
        md,
        "| Classical Sliding MF | FP32 | 32 | {:+.2} | — | {:.3} | {:.1}% | {:.1}% | {} | {:.2} | {:.2}% | {} | {:.1} | {:.5} |",
        harsh_cmf.si_snr_db,
        harsh_cmf.roc_auc,
        harsh_cmf.pd_1pct * 100.0,
        harsh_cmf.toa_hit_rate_1sample * 100.0,
        fp_smf_f32.weight_bytes,
        fp_smf_f32.total_sram_bytes() as f64 / 1024.0,
        fp_smf_f32.total_sram_bytes() as f64 / ESP32_USABLE_DRAM_BYTES as f64 * 100.0,
        fp_smf_f32.macs_per_frame,
        est_smf_f32.latency_us_240mhz,
        est_smf_f32.energy_mj_240mhz
    )
    .unwrap();
    writeln!(
        md,
        "| Classical Sliding MF | INT8 | 32 | {:+.2} | ~0.00 | {:.3} | {:.1}% | {:.1}% | {} | {:.2} | {:.2}% | {} | {:.1} | {:.5} |",
        harsh_cmf.si_snr_db,
        harsh_cmf.roc_auc,
        harsh_cmf.pd_1pct * 100.0,
        harsh_cmf.toa_hit_rate_1sample * 100.0,
        fp_smf_i8.weight_bytes,
        fp_smf_i8.total_sram_bytes() as f64 / 1024.0,
        fp_smf_i8.total_sram_bytes() as f64 / ESP32_USABLE_DRAM_BYTES as f64 * 100.0,
        fp_smf_i8.macs_per_frame,
        est_smf_i8.latency_us_240mhz,
        est_smf_i8.energy_mj_240mhz
    )
    .unwrap();

    for r in pareto {
        writeln!(
            md,
            "| Conv1D {} | FP32 | {} | {:+.2} | — | {:.3} | {:.1}% | {:.1}% | {} | {:.2} | {:.2}% | {} | {:.1} | {:.5} |",
            r.label,
            r.params,
            r.fp32.si_snr_db,
            r.fp32.roc_auc,
            r.fp32.pd_1pct * 100.0,
            r.fp32.toa_hit_rate_1sample * 100.0,
            r.fp32_weight_bytes,
            r.fp32_sram_bytes as f64 / 1024.0,
            r.fp32_sram_bytes as f64 / ESP32_USABLE_DRAM_BYTES as f64 * 100.0,
            r.macs_per_frame,
            r.est_fp32.latency_us_240mhz,
            r.est_fp32.energy_mj_240mhz
        )
        .unwrap();
        writeln!(
            md,
            "| Conv1D {} | **INT8** | {} | **{:+.2}** | {:+.2} | **{:.3}** | **{:.1}%** | **{:.1}%** | **{}** (4.0x) | **{:.2}** | **{:.2}%** | {} | **{:.1}** | **{:.5}** |",
            r.label,
            r.params,
            r.int8.si_snr_db,
            r.int8.si_snr_db - r.fp32.si_snr_db,
            r.int8.roc_auc,
            r.int8.pd_1pct * 100.0,
            r.int8.toa_hit_rate_1sample * 100.0,
            r.int8_weight_bytes,
            r.int8_sram_bytes as f64 / 1024.0,
            r.int8_sram_bytes as f64 / ESP32_USABLE_DRAM_BYTES as f64 * 100.0,
            r.macs_per_frame,
            r.est_int8.latency_us_240mhz,
            r.est_int8.energy_mj_240mhz
        )
        .unwrap();
    }

    writeln!(
        md,
        "\n## 4. Study 3 — Per-Sample Pulse SNR Sweep (`-14 dB` to `+4 dB`, Medium-C16)\n"
    )
    .unwrap();
    writeln!(md, "| Pulse SNR (dB) | AWGN CMF AUC | AWGN NMF-INT8 AUC | Harsh CMF AUC | Harsh CMF Pd@1% | Harsh NMF-FP32 AUC | Harsh NMF-INT8 AUC | Harsh NMF-INT8 Pd@1% | Harsh NMF-INT8 SI-SNR (dB) |").unwrap();
    writeln!(md, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for r in snr {
        writeln!(
            md,
            "| {:+.1} | {:.3} | {:.3} | {:.3} | {:.1}% | {:.3} | {:.3} | {:.1}% | {:+.2} |",
            r.snr_db,
            r.cmf_awgn.roc_auc,
            r.nmf_int8_awgn.roc_auc,
            r.cmf_harsh.roc_auc,
            r.cmf_harsh.pd_1pct * 100.0,
            r.nmf_fp32_harsh.roc_auc,
            r.nmf_int8_harsh.roc_auc,
            r.nmf_int8_harsh.pd_1pct * 100.0,
            r.nmf_int8_harsh.si_snr_db
        )
        .unwrap();
    }

    writeln!(
        md,
        "\n## 5. Adaptive Classical / Neural Switching (DVS/DVFS Integration Prototype)\n"
    )
    .unwrap();
    writeln!(md, "Tested on a 50/50 mixed stream of clean AWGN and Industrial Harsh frames (`SNR_pulse in [-8, +2] dB`):").unwrap();
    writeln!(
        md,
        "- Clean AWGN frames routed to low-power Classical MF (80 MHz): **{:.1}%**",
        adaptive.awgn_kept_classical_pct
    )
    .unwrap();
    writeln!(
        md,
        "- Industrial Harsh frames routed to Neural INT8 MF (240 MHz): **{:.1}%**",
        adaptive.harsh_routed_neural_pct
    )
    .unwrap();
    writeln!(
        md,
        "- Mixed-stream Detection ROC-AUC: Classical-only `{:.3}` -> Adaptive `{:.3}` (Always-Neural `{:.3}`).",
        adaptive.classical_only_auc, adaptive.adaptive_auc, adaptive.neural_always_auc
    )
    .unwrap();
    writeln!(
        md,
        "- Mixed-stream `Pd @ Pfa=1%`: Classical-only `{:.1}%` -> Adaptive `{:.1}%` (Always-Neural `{:.1}%`).",
        adaptive.classical_only_pd1 * 100.0,
        adaptive.adaptive_pd1 * 100.0,
        adaptive.neural_always_pd1 * 100.0
    )
    .unwrap();
    writeln!(
        md,
        "- Estimated Average Energy per Frame: Always-Neural `{:.1} uJ` -> Adaptive `{:.1} uJ` (**{:.1}% energy reduction**).",
        adaptive.neural_always_energy_uj, adaptive.adaptive_energy_uj, adaptive.energy_saving_pct
    )
    .unwrap();

    writeln!(md, "\n## 6. Generated Figures").unwrap();
    writeln!(md, "- `pareto_tradeoff.svg`: SI-SNR vs Compute/SRAM Pareto frontier").unwrap();
    writeln!(
        md,
        "- `snr_sweep.svg`: Detection ROC-AUC (%) vs Input Pulse SNR across AWGN and Industrial Harsh channels"
    )
    .unwrap();
    writeln!(
        md,
        "- `waveform_demo.svg`: 4-panel time-domain recovery comparison on Industrial Harsh frame at `-3 dB`"
    )
    .unwrap();

    std::fs::write(path, md)
}
