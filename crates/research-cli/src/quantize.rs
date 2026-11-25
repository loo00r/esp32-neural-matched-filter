//! Static INT8 Post-Training Quantization (PTQ) with outlier-aware percentile calibration.
//!
//! Produces an `OwnedModelInt8` backed by the exact `no_std` integer-only kernel
//! `neural_filter_core::conv1d_int8::NeuralFilterInt8`.

use neural_filter_core::{
    FRAME_LEN, KERNEL_SIZE,
    conv1d_f32::{ScratchpadF32, conv1d_forward_f32},
    conv1d_int8::{Conv1dLayerInt8, NeuralFilterInt8, ScratchpadInt8},
    quant::FixedPointScale,
};

use crate::{
    metrics::si_snr_db,
    signal::FrameSample,
    train::{OwnedConv1d, OwnedModelF32},
};

/// Owned static INT8 quantized Conv1D layer.
#[derive(Debug, Clone)]
pub struct OwnedConv1dInt8 {
    pub c_in: usize,
    pub c_out: usize,
    pub dilation: usize,
    pub weights: Vec<i8>,
    pub biases: Vec<i32>,
    pub scales: Vec<FixedPointScale>,
}

impl OwnedConv1dInt8 {
    pub fn as_view(&self) -> Conv1dLayerInt8<'_> {
        Conv1dLayerInt8 {
            c_in: self.c_in,
            c_out: self.c_out,
            dilation: self.dilation,
            weights: &self.weights,
            biases: &self.biases,
            scales: &self.scales,
        }
    }
}

/// Owned static INT8 3-layer dilated Conv1D model.
#[derive(Debug, Clone)]
pub struct OwnedModelInt8 {
    pub hidden_channels: usize,
    pub input_scale: f32,
    pub output_scale: f32,
    pub calibration_percentile: f32,
    pub layer1: OwnedConv1dInt8,
    pub layer2: OwnedConv1dInt8,
    pub layer3: OwnedConv1dInt8,
}

impl OwnedModelInt8 {
    pub fn as_view(&self) -> NeuralFilterInt8<'_> {
        NeuralFilterInt8 {
            hidden_channels: self.hidden_channels,
            input_scale: self.input_scale,
            output_scale: self.output_scale,
            layer1: self.layer1.as_view(),
            layer2: self.layer2.as_view(),
            layer3: self.layer3.as_view(),
        }
    }

    pub fn eval_si_snr(&self, dataset: &[FrameSample]) -> f32 {
        let view = self.as_view();
        let mut scratch = ScratchpadInt8::new();
        let mut out = [0.0_f32; FRAME_LEN];
        let mut sum = 0.0_f64;
        for sample in dataset {
            view.infer_f32_io(&sample.input, &mut out, &mut scratch);
            sum += si_snr_db(&out, &sample.target) as f64;
        }
        (sum / dataset.len().max(1) as f64) as f32
    }
}

/// Summary comparing calibrated percentile PTQ vs naive min-max PTQ.
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    pub model: OwnedModelInt8,
    pub minmax_cal_si_snr: f32,
    pub best_cal_si_snr: f32,
}

/// Quantizes `model_f32` using static per-channel symmetric weight quantization and
/// calibrated activation scales selected via percentile grid search on `calibration_set`.
pub fn quantize_static_int8(
    model_f32: &OwnedModelF32,
    calibration_set: &[FrameSample],
) -> QuantizationResult {
    assert!(!calibration_set.is_empty());

    // Collect activation magnitudes across the calibration dataset
    let c = model_f32.hidden_channels;
    let hidden_len = c * FRAME_LEN;
    let mut scratch = ScratchpadF32::new();
    let mut out = [0.0_f32; FRAME_LEN];

    let mut abs_in = Vec::with_capacity(calibration_set.len() * FRAME_LEN);
    let mut pos_a1 = Vec::with_capacity(calibration_set.len() * hidden_len);
    let mut pos_a2 = Vec::with_capacity(calibration_set.len() * hidden_len);
    let mut abs_out = Vec::with_capacity(calibration_set.len() * FRAME_LEN);

    for sample in calibration_set {
        abs_in.extend(sample.input.iter().map(|x| x.abs()));

        conv1d_forward_f32(
            &model_f32.layer1.as_view(),
            &sample.input,
            &mut scratch.buf_a[..hidden_len],
            true,
        );
        pos_a1.extend_from_slice(&scratch.buf_a[..hidden_len]);

        conv1d_forward_f32(
            &model_f32.layer2.as_view(),
            &scratch.buf_a[..hidden_len],
            &mut scratch.buf_b[..hidden_len],
            true,
        );
        pos_a2.extend_from_slice(&scratch.buf_b[..hidden_len]);

        conv1d_forward_f32(
            &model_f32.layer3.as_view(),
            &scratch.buf_b[..hidden_len],
            &mut out,
            false,
        );
        abs_out.extend(out.iter().map(|x| x.abs()));
    }

    abs_in.sort_by(|a, b| a.total_cmp(b));
    pos_a1.sort_by(|a, b| a.total_cmp(b));
    pos_a2.sort_by(|a, b| a.total_cmp(b));
    abs_out.sort_by(|a, b| a.total_cmp(b));

    let percentiles = [0.990_f32, 0.995, 0.998, 0.999, 0.9995, 1.0];
    let mut best_model: Option<OwnedModelInt8> = None;
    let mut best_si_snr = f32::NEG_INFINITY;
    let mut minmax_si_snr = f32::NEG_INFINITY;

    for &p in &percentiles {
        let s_in = percentile_sorted(&abs_in, p).max(1e-6) / 127.0;
        let s_a1 = percentile_sorted(&pos_a1, p).max(1e-6) / 255.0;
        let s_a2 = percentile_sorted(&pos_a2, p).max(1e-6) / 255.0;
        let s_out = percentile_sorted(&abs_out, p).max(1e-6) / 127.0;

        let candidate = build_quantized_model(model_f32, s_in, s_a1, s_a2, s_out, p);
        let score = candidate.eval_si_snr(calibration_set);

        if (p - 1.0).abs() < 1e-6 {
            minmax_si_snr = score;
        }
        if score > best_si_snr {
            best_si_snr = score;
            best_model = Some(candidate);
        }
    }

    QuantizationResult {
        model: best_model.expect("at least one percentile candidate"),
        minmax_cal_si_snr: minmax_si_snr,
        best_cal_si_snr: best_si_snr,
    }
}

fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 1.0;
    }
    let idx = (((sorted.len() - 1) as f64) * (p as f64)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn build_quantized_model(
    model_f32: &OwnedModelF32,
    s_in: f32,
    s_a1: f32,
    s_a2: f32,
    s_out: f32,
    percentile: f32,
) -> OwnedModelInt8 {
    OwnedModelInt8 {
        hidden_channels: model_f32.hidden_channels,
        input_scale: s_in,
        output_scale: s_out,
        calibration_percentile: percentile,
        layer1: quantize_layer(&model_f32.layer1, s_in, s_a1),
        layer2: quantize_layer(&model_f32.layer2, s_a1, s_a2),
        layer3: quantize_layer(&model_f32.layer3, s_a2, s_out),
    }
}

fn quantize_layer(layer: &OwnedConv1d, s_in: f32, s_out: f32) -> OwnedConv1dInt8 {
    let stride = layer.c_in * KERNEL_SIZE;
    let mut weights = vec![0_i8; layer.weights.len()];
    let mut biases = vec![0_i32; layer.c_out];
    let mut scales = vec![FixedPointScale::ZERO; layer.c_out];

    for oc in 0..layer.c_out {
        let slice = &layer.weights[oc * stride..(oc + 1) * stride];
        let max_w = slice.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        if max_w < 1e-9 {
            scales[oc] = FixedPointScale::ZERO;
            biases[oc] = 0;
            continue;
        }
        let s_w = max_w / 127.0;
        for i in 0..stride {
            let q = (slice[i] / s_w).round() as i32;
            weights[oc * stride + i] = q.clamp(-127, 127) as i8;
        }
        let acc_scale = (s_in as f64) * (s_w as f64);
        biases[oc] = ((layer.biases[oc] as f64) / acc_scale).round() as i32;
        scales[oc] = FixedPointScale::from_f64(acc_scale / (s_out as f64));
    }

    OwnedConv1dInt8 {
        c_in: layer.c_in,
        c_out: layer.c_out,
        dilation: layer.dilation,
        weights,
        biases,
        scales,
    }
}
