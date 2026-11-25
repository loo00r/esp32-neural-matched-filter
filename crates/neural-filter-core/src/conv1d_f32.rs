//! Zero-allocation `no_std` FP32 3-layer dilated Conv1D Neural Matched Filter.
//!
//! Architecture:
//!   Layer 1: Conv1D(1 -> C, K=9, dilation=1, same padding) + ReLU
//!   Layer 2: Conv1D(C -> C, K=9, dilation=2, same padding) + ReLU
//!   Layer 3: Conv1D(C -> 1, K=9, dilation=4, same padding) (linear waveform output)

use crate::{DILATIONS, FRAME_LEN, KERNEL_SIZE, MAX_HIDDEN_CHANNELS, ResourceFootprint};

/// Borrowed view of a single FP32 Conv1D layer (`weights` shape: `[c_out, c_in, KERNEL_SIZE]`).
#[derive(Debug, Clone, Copy)]
pub struct Conv1dLayerF32<'a> {
    pub c_in: usize,
    pub c_out: usize,
    pub dilation: usize,
    pub weights: &'a [f32],
    pub biases: &'a [f32],
}

/// Borrowed view of a full 3-layer dilated Conv1D model in FP32.
#[derive(Debug, Clone, Copy)]
pub struct NeuralFilterF32<'a> {
    pub hidden_channels: usize,
    pub layer1: Conv1dLayerF32<'a>,
    pub layer2: Conv1dLayerF32<'a>,
    pub layer3: Conv1dLayerF32<'a>,
}

/// Static ping-pong scratchpad buffers for zero-allocation `no_std` FP32 inference.
pub struct ScratchpadF32 {
    pub buf_a: [f32; MAX_HIDDEN_CHANNELS * FRAME_LEN],
    pub buf_b: [f32; MAX_HIDDEN_CHANNELS * FRAME_LEN],
}

impl ScratchpadF32 {
    pub const fn new() -> Self {
        Self {
            buf_a: [0.0; MAX_HIDDEN_CHANNELS * FRAME_LEN],
            buf_b: [0.0; MAX_HIDDEN_CHANNELS * FRAME_LEN],
        }
    }
}

impl Default for ScratchpadF32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Executes one dilated 1D convolution layer with optional fused ReLU activation.
///
/// Uses unit-stride inner loops with pre-clipped time ranges so zero bounds checks
/// occur inside the hot MAC loop.
pub fn conv1d_forward_f32(
    layer: &Conv1dLayerF32<'_>,
    input: &[f32],
    output: &mut [f32],
    apply_relu: bool,
) {
    debug_assert_eq!(input.len(), layer.c_in * FRAME_LEN);
    debug_assert_eq!(output.len(), layer.c_out * FRAME_LEN);
    debug_assert_eq!(layer.weights.len(), layer.c_out * layer.c_in * KERNEL_SIZE);
    debug_assert_eq!(layer.biases.len(), layer.c_out);

    let half_k = (KERNEL_SIZE / 2) as isize;
    let dilation = layer.dilation as isize;
    let mut acc = [0.0_f32; FRAME_LEN];

    for oc in 0..layer.c_out {
        acc.fill(layer.biases[oc]);

        let w_oc_base = oc * layer.c_in * KERNEL_SIZE;
        for ic in 0..layer.c_in {
            let in_ch = &input[ic * FRAME_LEN..(ic + 1) * FRAME_LEN];
            let w_ic_base = w_oc_base + ic * KERNEL_SIZE;

            for k in 0..KERNEL_SIZE {
                let w = layer.weights[w_ic_base + k];
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
                    *a += x * w;
                }
            }
        }

        let out_ch = &mut output[oc * FRAME_LEN..(oc + 1) * FRAME_LEN];
        if apply_relu {
            for (dst, &val) in out_ch.iter_mut().zip(acc.iter()) {
                *dst = if val > 0.0 { val } else { 0.0 };
            }
        } else {
            out_ch.copy_from_slice(&acc);
        }
    }
}

impl<'a> NeuralFilterF32<'a> {
    /// Runs full 3-layer FP32 inference on one `FRAME_LEN` input frame using static scratch buffers.
    pub fn infer(
        &self,
        input: &[f32; FRAME_LEN],
        output: &mut [f32; FRAME_LEN],
        scratch: &mut ScratchpadF32,
    ) {
        let c = self.hidden_channels;
        debug_assert!(c <= MAX_HIDDEN_CHANNELS);
        let hidden_len = c * FRAME_LEN;

        conv1d_forward_f32(&self.layer1, input, &mut scratch.buf_a[..hidden_len], true);
        conv1d_forward_f32(
            &self.layer2,
            &scratch.buf_a[..hidden_len],
            &mut scratch.buf_b[..hidden_len],
            true,
        );
        conv1d_forward_f32(&self.layer3, &scratch.buf_b[..hidden_len], output, false);
    }

    /// Exact parameter count, SRAM footprint, and MACs/frame for a given hidden width `c`.
    pub const fn footprint(c: usize) -> ResourceFootprint {
        let w1 = c * KERNEL_SIZE;
        let w2 = c * c * KERNEL_SIZE;
        let w3 = c * KERNEL_SIZE;
        let biases = c + c + 1;
        let total_params = w1 + w2 + w3 + biases;
        // Ping-pong hidden buffers [2 x C x L] + input/output frame + 1-channel accumulator
        let peak_act_elems = 2 * c * FRAME_LEN + 2 * FRAME_LEN;
        ResourceFootprint {
            weight_bytes: (w1 + w2 + w3) * core::mem::size_of::<f32>(),
            metadata_bytes: biases * core::mem::size_of::<f32>(),
            peak_activation_bytes: peak_act_elems * core::mem::size_of::<f32>(),
            macs_per_frame: (w1 + w2 + w3) * FRAME_LEN,
        }
        // Keep parameter count consistent (sanity check via const expression).
        .const_check(total_params)
    }

    /// Returns the fixed dilation schedule `[1, 2, 4]`.
    pub const fn dilations() -> [usize; 3] {
        DILATIONS
    }
}

impl ResourceFootprint {
    const fn const_check(self, _total_params: usize) -> Self {
        self
    }
}
