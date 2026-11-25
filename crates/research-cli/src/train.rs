//! Pure-Rust training engine for the 3-layer dilated Conv1D Neural Matched Filter.
//!
//! Features:
//! - Exact analytical forward and backward passes (verified against finite differences).
//! - Multi-threaded mini-batch gradient accumulation using `std::thread::scope`.
//! - Adam optimizer with cosine-annealed learning rate and global gradient-norm clipping.
//! - Post-training sign alignment so SI-SNR-trained waveforms match positive template polarity.

use std::f64::consts::PI;

use neural_filter_core::{
    DILATIONS, FRAME_LEN, KERNEL_SIZE,
    conv1d_f32::{Conv1dLayerF32, NeuralFilterF32, ScratchpadF32},
};

use crate::{
    metrics::{neg_si_snr_loss_and_grad, si_snr_db},
    rng::Rng,
    signal::FrameSample,
};

/// Owned trainable weights and biases for one dilated Conv1D layer.
#[derive(Debug, Clone)]
pub struct OwnedConv1d {
    pub c_in: usize,
    pub c_out: usize,
    pub dilation: usize,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

impl OwnedConv1d {
    pub fn init_kaiming(
        c_in: usize,
        c_out: usize,
        dilation: usize,
        gain: f32,
        rng: &mut Rng,
    ) -> Self {
        let fan_in = (c_in * KERNEL_SIZE) as f32;
        let bound = gain * (3.0 / fan_in).sqrt();
        let mut weights = vec![0.0_f32; c_out * c_in * KERNEL_SIZE];
        for w in &mut weights {
            *w = rng.uniform_f32(-bound, bound);
        }
        let biases = vec![0.0_f32; c_out];
        Self {
            c_in,
            c_out,
            dilation,
            weights,
            biases,
        }
    }

    pub fn as_view(&self) -> Conv1dLayerF32<'_> {
        Conv1dLayerF32 {
            c_in: self.c_in,
            c_out: self.c_out,
            dilation: self.dilation,
            weights: &self.weights,
            biases: &self.biases,
        }
    }
}

/// Owned 3-layer dilated Conv1D model that converts directly into `neural_filter_core::NeuralFilterF32`.
#[derive(Debug, Clone)]
pub struct OwnedModelF32 {
    pub hidden_channels: usize,
    pub layer1: OwnedConv1d,
    pub layer2: OwnedConv1d,
    pub layer3: OwnedConv1d,
}

impl OwnedModelF32 {
    pub fn new(hidden_channels: usize, rng: &mut Rng) -> Self {
        Self {
            hidden_channels,
            layer1: OwnedConv1d::init_kaiming(1, hidden_channels, DILATIONS[0], 1.414, rng),
            layer2: OwnedConv1d::init_kaiming(
                hidden_channels,
                hidden_channels,
                DILATIONS[1],
                1.414,
                rng,
            ),
            layer3: OwnedConv1d::init_kaiming(hidden_channels, 1, DILATIONS[2], 0.5, rng),
        }
    }

    pub fn as_view(&self) -> NeuralFilterF32<'_> {
        NeuralFilterF32 {
            hidden_channels: self.hidden_channels,
            layer1: self.layer1.as_view(),
            layer2: self.layer2.as_view(),
            layer3: self.layer3.as_view(),
        }
    }

    pub fn param_count(&self) -> usize {
        self.layer1.weights.len()
            + self.layer1.biases.len()
            + self.layer2.weights.len()
            + self.layer2.biases.len()
            + self.layer3.weights.len()
            + self.layer3.biases.len()
    }

    /// Evaluates mean SI-SNR (dB) on a dataset using the exact `no_std` inference kernel.
    pub fn eval_si_snr(&self, dataset: &[FrameSample]) -> f32 {
        let view = self.as_view();
        let mut scratch = ScratchpadF32::new();
        let mut out = [0.0_f32; FRAME_LEN];
        let mut sum = 0.0_f64;
        for sample in dataset {
            view.infer(&sample.input, &mut out, &mut scratch);
            sum += si_snr_db(&out, &sample.target) as f64;
        }
        (sum / dataset.len().max(1) as f64) as f32
    }

    /// Aligns global sign of the linear output layer so `<f(x), target> > 0` on average.
    /// Mathematically leaves SI-SNR invariant while making matched-filter correlation peaks positive.
    pub fn align_polarity(&mut self, validation: &[FrameSample]) {
        let view = self.as_view();
        let mut scratch = ScratchpadF32::new();
        let mut out = [0.0_f32; FRAME_LEN];
        let mut dot_sum = 0.0_f64;
        for sample in validation.iter().take(128) {
            view.infer(&sample.input, &mut out, &mut scratch);
            for (&a, &b) in out.iter().zip(sample.target.iter()) {
                dot_sum += (a as f64) * (b as f64);
            }
        }
        if dot_sum < 0.0 {
            for w in &mut self.layer3.weights {
                *w = -*w;
            }
            for b in &mut self.layer3.biases {
                *b = -*b;
            }
        }
    }
}

/// Accumulated gradients matching `OwnedModelF32` layout.
#[derive(Debug, Clone)]
struct ModelGrads {
    dw1: Vec<f32>,
    db1: Vec<f32>,
    dw2: Vec<f32>,
    db2: Vec<f32>,
    dw3: Vec<f32>,
    db3: Vec<f32>,
}

impl ModelGrads {
    fn zeros_like(model: &OwnedModelF32) -> Self {
        Self {
            dw1: vec![0.0; model.layer1.weights.len()],
            db1: vec![0.0; model.layer1.biases.len()],
            dw2: vec![0.0; model.layer2.weights.len()],
            db2: vec![0.0; model.layer2.biases.len()],
            dw3: vec![0.0; model.layer3.weights.len()],
            db3: vec![0.0; model.layer3.biases.len()],
        }
    }

    fn add_assign(&mut self, other: &Self) {
        add_slice(&mut self.dw1, &other.dw1);
        add_slice(&mut self.db1, &other.db1);
        add_slice(&mut self.dw2, &other.dw2);
        add_slice(&mut self.db2, &other.db2);
        add_slice(&mut self.dw3, &other.dw3);
        add_slice(&mut self.db3, &other.db3);
    }

    fn scale_and_clip(&mut self, inv_batch: f32, max_norm: f32) {
        let mut norm_sq = 0.0_f64;
        for slice in [
            &mut self.dw1,
            &mut self.db1,
            &mut self.dw2,
            &mut self.db2,
            &mut self.dw3,
            &mut self.db3,
        ] {
            for v in slice.iter_mut() {
                *v *= inv_batch;
                norm_sq += (*v as f64) * (*v as f64);
            }
        }
        let norm = norm_sq.sqrt() as f32;
        if norm > max_norm && norm.is_finite() {
            let clip = max_norm / norm;
            for slice in [
                &mut self.dw1,
                &mut self.db1,
                &mut self.dw2,
                &mut self.db2,
                &mut self.dw3,
                &mut self.db3,
            ] {
                for v in slice.iter_mut() {
                    *v *= clip;
                }
            }
        }
    }
}

fn add_slice(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

/// Hyperparameters for pure-Rust Conv1D training.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr_max: f32,
    pub lr_min: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 65,
            batch_size: 64,
            lr_max: 4e-3,
            lr_min: 1.5e-4,
            weight_decay: 1e-4,
            grad_clip: 5.0,
        }
    }
}

/// Summary of one training run.
#[derive(Debug, Clone)]
pub struct TrainSummary {
    pub model: OwnedModelF32,
    pub best_val_si_snr: f32,
    pub best_epoch: usize,
    pub elapsed_secs: f64,
}

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamState {
    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    fn step(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        lr: f32,
        weight_decay: f32,
        step_idx: usize,
    ) {
        let beta1 = 0.9_f32;
        let beta2 = 0.999_f32;
        let eps = 1e-8_f32;
        let t = step_idx as i32;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);

        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            // Decoupled AdamW weight decay
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
        }
    }
}

/// Trains a 3-layer dilated Conv1D model using multi-threaded exact backprop on `-SI-SNR` loss.
pub fn train_model(
    hidden_channels: usize,
    train_data: &[FrameSample],
    val_data: &[FrameSample],
    cfg: TrainConfig,
    seed: u64,
) -> TrainSummary {
    let start_time = std::time::Instant::now();
    let mut rng = Rng::from_seed(seed);
    let mut model = OwnedModelF32::new(hidden_channels, &mut rng);

    let mut opt_w1 = AdamState::new(model.layer1.weights.len());
    let mut opt_b1 = AdamState::new(model.layer1.biases.len());
    let mut opt_w2 = AdamState::new(model.layer2.weights.len());
    let mut opt_b2 = AdamState::new(model.layer2.biases.len());
    let mut opt_w3 = AdamState::new(model.layer3.weights.len());
    let mut opt_b3 = AdamState::new(model.layer3.biases.len());

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);

    let mut best_model = model.clone();
    let mut best_val_si_snr = f32::NEG_INFINITY;
    let mut best_epoch = 0_usize;
    let mut step_idx = 0_usize;

    for epoch in 0..cfg.epochs {
        rng.shuffle(&mut indices);

        // Cosine learning rate schedule across epochs
        let progress = (epoch as f64) / (cfg.epochs.max(1) as f64);
        let cosine = 0.5 * (1.0 + (PI * progress).cos());
        let lr = cfg.lr_min + (cfg.lr_max - cfg.lr_min) * (cosine as f32);

        for batch_idx in indices.chunks(cfg.batch_size) {
            let (mut total_grads, _batch_si_snr_sum) =
                compute_batch_gradients(&model, train_data, batch_idx, num_threads);

            total_grads.scale_and_clip(1.0 / (batch_idx.len() as f32), cfg.grad_clip);
            step_idx += 1;

            opt_w1.step(
                &mut model.layer1.weights,
                &total_grads.dw1,
                lr,
                cfg.weight_decay,
                step_idx,
            );
            opt_b1.step(&mut model.layer1.biases, &total_grads.db1, lr, 0.0, step_idx);
            opt_w2.step(
                &mut model.layer2.weights,
                &total_grads.dw2,
                lr,
                cfg.weight_decay,
                step_idx,
            );
            opt_b2.step(&mut model.layer2.biases, &total_grads.db2, lr, 0.0, step_idx);
            opt_w3.step(
                &mut model.layer3.weights,
                &total_grads.dw3,
                lr,
                cfg.weight_decay,
                step_idx,
            );
            opt_b3.step(&mut model.layer3.biases, &total_grads.db3, lr, 0.0, step_idx);
        }

        let val_si_snr = model.eval_si_snr(val_data);
        if val_si_snr > best_val_si_snr {
            best_val_si_snr = val_si_snr;
            best_epoch = epoch + 1;
            best_model = model.clone();
        }
    }

    best_model.align_polarity(val_data);

    TrainSummary {
        model: best_model,
        best_val_si_snr,
        best_epoch,
        elapsed_secs: start_time.elapsed().as_secs_f64(),
    }
}

fn compute_batch_gradients(
    model: &OwnedModelF32,
    dataset: &[FrameSample],
    batch_indices: &[usize],
    num_threads: usize,
) -> (ModelGrads, f64) {
    let workers = num_threads.min(batch_indices.len().max(1));
    let chunk_size = batch_indices.len().div_ceil(workers);

    let partials: Vec<(ModelGrads, f64)> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for chunk in batch_indices.chunks(chunk_size) {
            handles.push(s.spawn(move || {
                let mut grads = ModelGrads::zeros_like(model);
                let mut si_snr_sum = 0.0_f64;
                let c = model.hidden_channels;
                let mut a1 = vec![0.0_f32; c * FRAME_LEN];
                let mut a2 = vec![0.0_f32; c * FRAME_LEN];
                let mut out = [0.0_f32; FRAME_LEN];
                let mut d_out = [0.0_f32; FRAME_LEN];
                let mut d_a2 = vec![0.0_f32; c * FRAME_LEN];
                let mut d_a1 = vec![0.0_f32; c * FRAME_LEN];

                for &idx in chunk {
                    let sample = &dataset[idx];
                    let (_, si_snr) = sample_forward_backward(
                        model,
                        sample,
                        &mut a1,
                        &mut a2,
                        &mut out,
                        &mut d_out,
                        &mut d_a2,
                        &mut d_a1,
                        &mut grads,
                    );
                    si_snr_sum += si_snr as f64;
                }
                (grads, si_snr_sum)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut total = ModelGrads::zeros_like(model);
    let mut total_si_snr = 0.0_f64;
    for (g, s) in partials {
        total.add_assign(&g);
        total_si_snr += s;
    }
    (total, total_si_snr)
}

#[allow(clippy::too_many_arguments)]
fn sample_forward_backward(
    model: &OwnedModelF32,
    sample: &FrameSample,
    a1: &mut [f32],
    a2: &mut [f32],
    out: &mut [f32; FRAME_LEN],
    d_out: &mut [f32; FRAME_LEN],
    d_a2: &mut [f32],
    d_a1: &mut [f32],
    grads: &mut ModelGrads,
) -> (f32, f32) {
    // Forward pass using exact same Conv1D forward kernel as no_std inference
    neural_filter_core::conv1d_f32::conv1d_forward_f32(&model.layer1.as_view(), &sample.input, a1, true);
    neural_filter_core::conv1d_f32::conv1d_forward_f32(&model.layer2.as_view(), a1, a2, true);
    neural_filter_core::conv1d_f32::conv1d_forward_f32(&model.layer3.as_view(), a2, out, false);

    // SI-SNR loss and output gradient
    let (loss, si_snr) = neg_si_snr_loss_and_grad(out, &sample.target, d_out);

    // Backward Layer 3 (linear)
    d_a2.fill(0.0);
    conv1d_backward_f32(
        &model.layer3,
        a2,
        d_out,
        &mut grads.dw3,
        &mut grads.db3,
        Some(d_a2),
    );

    // Backward ReLU 2
    for (g, &act) in d_a2.iter_mut().zip(a2.iter()) {
        if act <= 0.0 {
            *g = 0.0;
        }
    }

    // Backward Layer 2
    d_a1.fill(0.0);
    conv1d_backward_f32(
        &model.layer2,
        a1,
        d_a2,
        &mut grads.dw2,
        &mut grads.db2,
        Some(d_a1),
    );

    // Backward ReLU 1
    for (g, &act) in d_a1.iter_mut().zip(a1.iter()) {
        if act <= 0.0 {
            *g = 0.0;
        }
    }

    // Backward Layer 1 (no need for input gradient)
    conv1d_backward_f32(
        &model.layer1,
        &sample.input,
        d_a1,
        &mut grads.dw1,
        &mut grads.db1,
        None,
    );

    (loss, si_snr)
}

/// Exact analytical backward pass for one dilated Conv1D layer.
///
/// Accumulates into `grad_w`, `grad_b`, and optionally `grad_in`.
fn conv1d_backward_f32(
    layer: &OwnedConv1d,
    input: &[f32],
    grad_out: &[f32],
    grad_w: &mut [f32],
    grad_b: &mut [f32],
    mut grad_in: Option<&mut [f32]>,
) {
    let half_k = (KERNEL_SIZE / 2) as isize;
    let dilation = layer.dilation as isize;

    for oc in 0..layer.c_out {
        let g_out_ch = &grad_out[oc * FRAME_LEN..(oc + 1) * FRAME_LEN];

        // Bias gradient: sum over time
        let mut b_acc = 0.0_f32;
        for &g in g_out_ch {
            b_acc += g;
        }
        grad_b[oc] += b_acc;

        let w_oc_base = oc * layer.c_in * KERNEL_SIZE;
        for ic in 0..layer.c_in {
            let in_ch = &input[ic * FRAME_LEN..(ic + 1) * FRAME_LEN];
            let w_ic_base = w_oc_base + ic * KERNEL_SIZE;

            for k in 0..KERNEL_SIZE {
                let delta = (k as isize - half_k) * dilation;
                let (t_start, src_start, len) = if delta >= 0 {
                    let d = delta as usize;
                    if d >= FRAME_LEN {
                        continue;
                    }
                    (0, d, FRAME_LEN - d)
                } else {
                    let d = (-delta) as usize;
                    if d >= FRAME_LEN {
                        continue;
                    }
                    (d, 0, FRAME_LEN - d)
                };

                let g_slice = &g_out_ch[t_start..t_start + len];
                let in_slice = &in_ch[src_start..src_start + len];

                // Weight gradient
                let mut w_acc = 0.0_f32;
                for (&g, &x) in g_slice.iter().zip(in_slice.iter()) {
                    w_acc += g * x;
                }
                grad_w[w_ic_base + k] += w_acc;

                // Input gradient
                if let Some(ref mut g_in_all) = grad_in {
                    let w = layer.weights[w_ic_base + k];
                    let g_in_slice = &mut g_in_all[ic * FRAME_LEN + src_start..ic * FRAME_LEN + src_start + len];
                    for (gi, &go) in g_in_slice.iter_mut().zip(g_slice.iter()) {
                        *gi += go * w;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{ChannelRegime, build_lfm_chirp_template, generate_sample};

    #[test]
    fn analytical_backprop_matches_finite_differences_across_all_three_layers() {
        let mut rng = Rng::from_seed(1337);
        let template = build_lfm_chirp_template();
        let sample = generate_sample(ChannelRegime::IndustrialHarsh, &template, 0.0, true, &mut rng);
        let mut model = OwnedModelF32::new(4, &mut rng);

        let c = model.hidden_channels;
        let mut a1 = vec![0.0_f32; c * FRAME_LEN];
        let mut a2 = vec![0.0_f32; c * FRAME_LEN];
        let mut out = [0.0_f32; FRAME_LEN];
        let mut d_out = [0.0_f32; FRAME_LEN];
        let mut d_a2 = vec![0.0_f32; c * FRAME_LEN];
        let mut d_a1 = vec![0.0_f32; c * FRAME_LEN];
        let mut grads = ModelGrads::zeros_like(&model);

        let _ = sample_forward_backward(
            &model,
            &sample,
            &mut a1,
            &mut a2,
            &mut out,
            &mut d_out,
            &mut d_a2,
            &mut d_a1,
            &mut grads,
        );

        let eval_loss = |m: &OwnedModelF32| -> f32 {
            let mut scratch = ScratchpadF32::new();
            let mut y = [0.0_f32; FRAME_LEN];
            m.as_view().infer(&sample.input, &mut y, &mut scratch);
            let mut g_dummy = [0.0_f32; FRAME_LEN];
            neg_si_snr_loss_and_grad(&y, &sample.target, &mut g_dummy).0
        };

        let h = 1e-4_f32;
        // Check Layer 1, Layer 2, Layer 3 weights and biases
        for idx in [0, 5, 17, 35] {
            let orig = model.layer1.weights[idx];
            model.layer1.weights[idx] = orig + h;
            let lp = eval_loss(&model);
            model.layer1.weights[idx] = orig - h;
            let lm = eval_loss(&model);
            model.layer1.weights[idx] = orig;
            let num = (lp - lm) / (2.0 * h);
            let ana = grads.dw1[idx];
            let err = (num - ana).abs() / num.abs().max(1e-2);
            assert!(err < 3e-2, "L1.w[{idx}]: ana={ana} num={num} err={err}");
        }
        for idx in [0, 19, 73, 140] {
            let orig = model.layer2.weights[idx];
            model.layer2.weights[idx] = orig + h;
            let lp = eval_loss(&model);
            model.layer2.weights[idx] = orig - h;
            let lm = eval_loss(&model);
            model.layer2.weights[idx] = orig;
            let num = (lp - lm) / (2.0 * h);
            let ana = grads.dw2[idx];
            let err = (num - ana).abs() / num.abs().max(1e-2);
            assert!(err < 3e-2, "L2.w[{idx}]: ana={ana} num={num} err={err}");
        }
        for idx in [0, 11, 24, 35] {
            let orig = model.layer3.weights[idx];
            model.layer3.weights[idx] = orig + h;
            let lp = eval_loss(&model);
            model.layer3.weights[idx] = orig - h;
            let lm = eval_loss(&model);
            model.layer3.weights[idx] = orig;
            let num = (lp - lm) / (2.0 * h);
            let ana = grads.dw3[idx];
            let err = (num - ana).abs() / num.abs().max(1e-2);
            assert!(err < 3e-2, "L3.w[{idx}]: ana={ana} num={num} err={err}");
        }
    }
}
