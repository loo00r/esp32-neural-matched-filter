//! Zero-allocation `no_std` static INT8 integer-only 3-layer dilated Conv1D inference.
//!
//! Quantization scheme (zero floating-point operations inside inference):
//! - Input frame: symmetric signed `i8` (`[-127, 127]`, `zp = 0`, scale `input_scale`).
//! - Weights: per-output-channel symmetric signed `i8` (`[-127, 127]`, `zp = 0`).
//! - Biases: pre-quantized `i32` (`round(b / (s_in * s_w[oc]))`).
//! - Post-ReLU hidden activations (after layers 1 & 2): unsigned `u8` (`[0, 255]`, `zp = 0`).
//!   Using `u8` for non-negative ReLU activations preserves a full extra bit of precision
//!   (256 levels instead of 128) while keeping `zero_point = 0`, so zero-padding is free
//!   and the inner MAC loop is strictly `acc += (x as i32) * (w as i32)`.
//! - Final waveform output: symmetric signed `i8` (`[-127, 127]`, `zp = 0`, scale `output_scale`).

use crate::{
    FRAME_LEN, KERNEL_SIZE, MAX_HIDDEN_CHANNELS, ResourceFootprint,
    quant::{FixedPointScale, dequantize_frame_i8, quantize_frame_i8},
};

/// Borrowed static INT8 layer parameters (`weights` shape: `[c_out, c_in, KERNEL_SIZE]`).
#[derive(Debug, Clone, Copy)]
pub struct Conv1dLayerInt8<'a> {
    pub c_in: usize,
    pub c_out: usize,
    pub dilation: usize,
    pub weights: &'a [i8],
    pub biases: &'a [i32],
    /// Per-output-channel fixed-point requantization scales `M[oc] = (s_in * s_w[oc]) / s_out`.
    pub scales: &'a [FixedPointScale],
}

/// Borrowed view of a full 3-layer static INT8 Neural Matched Filter.
#[derive(Debug, Clone, Copy)]
pub struct NeuralFilterInt8<'a> {
    pub hidden_channels: usize,
    pub input_scale: f32,
    pub output_scale: f32,
    pub layer1: Conv1dLayerInt8<'a>,
    pub layer2: Conv1dLayerInt8<'a>,
    pub layer3: Conv1dLayerInt8<'a>,
}

/// Static ping-pong scratchpad buffers for zero-allocation `no_std` INT8 inference.
///
/// Uses 4x less SRAM than `ScratchpadF32` (1 byte per hidden activation instead of 4).
pub struct ScratchpadInt8 {
    pub buf_a: [u8; MAX_HIDDEN_CHANNELS * FRAME_LEN],
    pub buf_b: [u8; MAX_HIDDEN_CHANNELS * FRAME_LEN],
}

impl ScratchpadInt8 {
    pub const fn new() -> Self {
        Self {
            buf_a: [0; MAX_HIDDEN_CHANNELS * FRAME_LEN],
            buf_b: [0; MAX_HIDDEN_CHANNELS * FRAME_LEN],
        }
    }
}

impl Default for ScratchpadInt8 {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer 1 kernel: signed `i8` input -> fused ReLU unsigned `u8` hidden activations.
pub fn conv1d_i8_to_u8_relu(layer: &Conv1dLayerInt8<'_>, input: &[i8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), layer.c_in * FRAME_LEN);
    debug_assert_eq!(output.len(), layer.c_out * FRAME_LEN);

    let half_k = (KERNEL_SIZE / 2) as isize;
    let dilation = layer.dilation as isize;
    let mut acc = [0_i32; FRAME_LEN];

    for oc in 0..layer.c_out {
        acc.fill(layer.biases[oc]);

        let w_oc_base = oc * layer.c_in * KERNEL_SIZE;
        for ic in 0..layer.c_in {
            let in_ch = &input[ic * FRAME_LEN..(ic + 1) * FRAME_LEN];
            let w_ic_base = w_oc_base + ic * KERNEL_SIZE;

            for k in 0..KERNEL_SIZE {
                let w = layer.weights[w_ic_base + k] as i32;
                if w == 0 {
                    continue;
                }
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

                let acc_slice = &mut acc[t_start..t_start + len];
                let in_slice = &in_ch[src_start..src_start + len];
                for (a, &x) in acc_slice.iter_mut().zip(in_slice.iter()) {
                    *a += (x as i32) * w;
                }
            }
        }

        let scale = layer.scales[oc];
        let out_ch = &mut output[oc * FRAME_LEN..(oc + 1) * FRAME_LEN];
        for (dst, &val) in out_ch.iter_mut().zip(acc.iter()) {
            *dst = scale.apply_relu_u8(val);
        }
    }
}

/// Layer 2 kernel: unsigned `u8` hidden input -> fused ReLU unsigned `u8` hidden output.
pub fn conv1d_u8_to_u8_relu(layer: &Conv1dLayerInt8<'_>, input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len(), layer.c_in * FRAME_LEN);
    debug_assert_eq!(output.len(), layer.c_out * FRAME_LEN);

    let half_k = (KERNEL_SIZE / 2) as isize;
    let dilation = layer.dilation as isize;
    let mut acc = [0_i32; FRAME_LEN];

    for oc in 0..layer.c_out {
        acc.fill(layer.biases[oc]);

        let w_oc_base = oc * layer.c_in * KERNEL_SIZE;
        for ic in 0..layer.c_in {
            let in_ch = &input[ic * FRAME_LEN..(ic + 1) * FRAME_LEN];
            let w_ic_base = w_oc_base + ic * KERNEL_SIZE;

            for k in 0..KERNEL_SIZE {
                let w = layer.weights[w_ic_base + k] as i32;
                if w == 0 {
                    continue;
                }
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

                let acc_slice = &mut acc[t_start..t_start + len];
                let in_slice = &in_ch[src_start..src_start + len];
                for (a, &x) in acc_slice.iter_mut().zip(in_slice.iter()) {
                    *a += (x as i32) * w;
                }
            }
        }

        let scale = layer.scales[oc];
        let out_ch = &mut output[oc * FRAME_LEN..(oc + 1) * FRAME_LEN];
        for (dst, &val) in out_ch.iter_mut().zip(acc.iter()) {
            *dst = scale.apply_relu_u8(val);
        }
    }
}

/// Layer 3 kernel: unsigned `u8` hidden input -> linear signed `i8` output waveform.
pub fn conv1d_u8_to_i8_linear(layer: &Conv1dLayerInt8<'_>, input: &[u8], output: &mut [i8]) {
    debug_assert_eq!(input.len(), layer.c_in * FRAME_LEN);
    debug_assert_eq!(output.len(), layer.c_out * FRAME_LEN);

    let half_k = (KERNEL_SIZE / 2) as isize;
    let dilation = layer.dilation as isize;
    let mut acc = [0_i32; FRAME_LEN];

    for oc in 0..layer.c_out {
        acc.fill(layer.biases[oc]);

        let w_oc_base = oc * layer.c_in * KERNEL_SIZE;
        for ic in 0..layer.c_in {
            let in_ch = &input[ic * FRAME_LEN..(ic + 1) * FRAME_LEN];
            let w_ic_base = w_oc_base + ic * KERNEL_SIZE;

            for k in 0..KERNEL_SIZE {
                let w = layer.weights[w_ic_base + k] as i32;
                if w == 0 {
                    continue;
                }
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

                let acc_slice = &mut acc[t_start..t_start + len];
                let in_slice = &in_ch[src_start..src_start + len];
                for (a, &x) in acc_slice.iter_mut().zip(in_slice.iter()) {
                    *a += (x as i32) * w;
                }
            }
        }

        let scale = layer.scales[oc];
        let out_ch = &mut output[oc * FRAME_LEN..(oc + 1) * FRAME_LEN];
        for (dst, &val) in out_ch.iter_mut().zip(acc.iter()) {
            *dst = scale.apply_sym_i8(val);
        }
    }
}

impl<'a> NeuralFilterInt8<'a> {
    /// Runs strictly integer-only 3-layer inference on a pre-quantized `i8` frame.
    pub fn infer_quantized(
        &self,
        input_q: &[i8; FRAME_LEN],
        output_q: &mut [i8; FRAME_LEN],
        scratch: &mut ScratchpadInt8,
    ) {
        let c = self.hidden_channels;
        debug_assert!(c <= MAX_HIDDEN_CHANNELS);
        let hidden_len = c * FRAME_LEN;

        conv1d_i8_to_u8_relu(&self.layer1, input_q, &mut scratch.buf_a[..hidden_len]);
        conv1d_u8_to_u8_relu(
            &self.layer2,
            &scratch.buf_a[..hidden_len],
            &mut scratch.buf_b[..hidden_len],
        );
        conv1d_u8_to_i8_linear(&self.layer3, &scratch.buf_b[..hidden_len], output_q);
    }

    /// Convenience wrapper that quantizes `f32` input, runs integer-only inference, and dequantizes output.
    pub fn infer_f32_io(
        &self,
        input: &[f32; FRAME_LEN],
        output: &mut [f32; FRAME_LEN],
        scratch: &mut ScratchpadInt8,
    ) {
        let mut input_q = [0_i8; FRAME_LEN];
        let mut output_q = [0_i8; FRAME_LEN];
        quantize_frame_i8(input, self.input_scale, &mut input_q);
        self.infer_quantized(&input_q, &mut output_q, scratch);
        dequantize_frame_i8(&output_q, self.output_scale, output);
    }

    /// Exact static INT8 weight, metadata, SRAM activation, and MAC accounting for hidden width `c`.
    pub const fn footprint(c: usize) -> ResourceFootprint {
        let w1 = c * KERNEL_SIZE;
        let w2 = c * c * KERNEL_SIZE;
        let w3 = c * KERNEL_SIZE;
        let out_channels_total = c + c + 1;
        // Per output channel: 4-byte i32 bias + 5-byte (8-byte aligned) FixedPointScale + 2 f32 I/O scales
        let metadata_bytes = out_channels_total
            * (core::mem::size_of::<i32>() + core::mem::size_of::<FixedPointScale>())
            + 2 * core::mem::size_of::<f32>();
        // Ping-pong hidden buffers [2 x C x L] bytes + input/output i8 frames + 1-channel i32 accumulator (512 B)
        let peak_activation_bytes =
            2 * c * FRAME_LEN + 2 * FRAME_LEN + FRAME_LEN * core::mem::size_of::<i32>();

        ResourceFootprint {
            weight_bytes: w1 + w2 + w3,
            metadata_bytes,
            peak_activation_bytes,
            macs_per_frame: (w1 + w2 + w3) * FRAME_LEN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv1d_f32::{Conv1dLayerF32, NeuralFilterF32, ScratchpadF32};

    #[test]
    fn int8_kernel_tracks_fp32_kernel_on_synthetic_weights() {
        const C: usize = 4;
        let mut w1_f = [0.0_f32; C * KERNEL_SIZE];
        let mut b1_f = [0.0_f32; C];
        let mut w2_f = [0.0_f32; C * C * KERNEL_SIZE];
        let mut b2_f = [0.0_f32; C];
        let mut w3_f = [0.0_f32; C * KERNEL_SIZE];
        let b3_f = [0.02_f32; 1];

        for (i, v) in w1_f.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin()) * 0.4;
        }
        for (i, v) in b1_f.iter_mut().enumerate() {
            *v = 0.05 * (i as f32);
        }
        for (i, v) in w2_f.iter_mut().enumerate() {
            *v = ((i as f32 * 0.23).cos()) * 0.25;
        }
        for (i, v) in b2_f.iter_mut().enumerate() {
            *v = 0.02 * (i as f32);
        }
        for (i, v) in w3_f.iter_mut().enumerate() {
            *v = ((i as f32 * 0.51).sin()) * 0.3;
        }

        let model_f = NeuralFilterF32 {
            hidden_channels: C,
            layer1: Conv1dLayerF32 {
                c_in: 1,
                c_out: C,
                dilation: 1,
                weights: &w1_f,
                biases: &b1_f,
            },
            layer2: Conv1dLayerF32 {
                c_in: C,
                c_out: C,
                dilation: 2,
                weights: &w2_f,
                biases: &b2_f,
            },
            layer3: Conv1dLayerF32 {
                c_in: C,
                c_out: 1,
                dilation: 4,
                weights: &w3_f,
                biases: &b3_f,
            },
        };

        let mut input = [0.0_f32; FRAME_LEN];
        for (t, x) in input.iter_mut().enumerate() {
            *x = (t as f32 * 0.19).sin();
        }
        let mut out_f = [0.0_f32; FRAME_LEN];
        let mut scratch_f = ScratchpadF32::new();
        model_f.infer(&input, &mut out_f, &mut scratch_f);

        // Simple static quantization with exact observed layer ranges
        let s_in = 1.0_f32 / 127.0;
        let max_a1 = scratch_f.buf_a[..C * FRAME_LEN]
            .iter()
            .fold(1e-6_f32, |m, &x| if x > m { x } else { m });
        let s_a1 = max_a1 / 255.0;
        let max_a2 = scratch_f.buf_b[..C * FRAME_LEN]
            .iter()
            .fold(1e-6_f32, |m, &x| if x > m { x } else { m });
        let s_a2 = max_a2 / 255.0;
        let max_out = out_f
            .iter()
            .fold(1e-6_f32, |m, &x| if x.abs() > m { x.abs() } else { m });
        let s_out = max_out / 127.0;

        fn quant_layer<const W: usize, const O: usize>(
            c_in: usize,
            c_out: usize,
            w_f: &[f32; W],
            b_f: &[f32; O],
            s_in: f32,
            s_out: f32,
        ) -> ([i8; W], [i32; O], [FixedPointScale; O]) {
            let mut w_q = [0_i8; W];
            let mut b_q = [0_i32; O];
            let mut scales = [FixedPointScale::ZERO; O];
            let stride = c_in * KERNEL_SIZE;
            for oc in 0..c_out {
                let slice = &w_f[oc * stride..(oc + 1) * stride];
                let max_w = slice
                    .iter()
                    .fold(1e-7_f32, |m, &x| if x.abs() > m { x.abs() } else { m });
                let s_w = max_w / 127.0;
                for i in 0..stride {
                    let q = slice[i] / s_w;
                    let r = if q >= 0.0 {
                        (q + 0.5) as i32
                    } else {
                        (q - 0.5) as i32
                    };
                    w_q[oc * stride + i] = r.clamp(-127, 127) as i8;
                }
                let b_scaled = b_f[oc] / (s_in * s_w);
                b_q[oc] = if b_scaled >= 0.0 {
                    (b_scaled + 0.5) as i32
                } else {
                    (b_scaled - 0.5) as i32
                };
                scales[oc] = FixedPointScale::from_f64((s_in as f64) * (s_w as f64) / (s_out as f64));
            }
            (w_q, b_q, scales)
        }

        let (w1_q, b1_q, sc1) = quant_layer(1, C, &w1_f, &b1_f, s_in, s_a1);
        let (w2_q, b2_q, sc2) = quant_layer(C, C, &w2_f, &b2_f, s_a1, s_a2);
        let (w3_q, b3_q, sc3) = quant_layer(C, 1, &w3_f, &b3_f, s_a2, s_out);

        let model_q = NeuralFilterInt8 {
            hidden_channels: C,
            input_scale: s_in,
            output_scale: s_out,
            layer1: Conv1dLayerInt8 {
                c_in: 1,
                c_out: C,
                dilation: 1,
                weights: &w1_q,
                biases: &b1_q,
                scales: &sc1,
            },
            layer2: Conv1dLayerInt8 {
                c_in: C,
                c_out: C,
                dilation: 2,
                weights: &w2_q,
                biases: &b2_q,
                scales: &sc2,
            },
            layer3: Conv1dLayerInt8 {
                c_in: C,
                c_out: 1,
                dilation: 4,
                weights: &w3_q,
                biases: &b3_q,
                scales: &sc3,
            },
        };

        let mut out_q = [0.0_f32; FRAME_LEN];
        let mut scratch_q = ScratchpadInt8::new();
        model_q.infer_f32_io(&input, &mut out_q, &mut scratch_q);

        let max_err = out_f
            .iter()
            .zip(out_q.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_err < 0.05 * max_out + 1e-3,
            "INT8 output diverged from FP32: max_err={max_err}, max_out={max_out}"
        );
    }
}
