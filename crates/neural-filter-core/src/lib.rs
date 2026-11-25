#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![forbid(unsafe_code)]

//! Core `no_std`, zero-allocation signal processing and 3-layer dilated Conv1D
//! Neural Matched Filter inference kernels targeting ESP32-WROOM-32 IoT nodes.
//!
//! This crate is shared bit-for-bit between host evaluation (`research-cli`)
//! and bare-metal ESP32 firmware (`esp32-benchmark`).

pub mod adaptive;
pub mod classical;
pub mod conv1d_f32;
pub mod conv1d_int8;
pub mod quant;

/// Fixed baseband processing frame length (samples).
/// At f_s = 16 kHz, 128 samples corresponds to an 8 ms IoT frame.
pub const FRAME_LEN: usize = 128;

/// Known IoT preamble / LFM chirp template length (samples).
/// At f_s = 16 kHz, 32 samples corresponds to a 2 ms chirp pulse.
pub const TEMPLATE_LEN: usize = 32;

/// Number of valid sliding matched-filter correlation lags (`FRAME_LEN - TEMPLATE_LEN + 1`).
pub const VALID_CORR_LEN: usize = FRAME_LEN - TEMPLATE_LEN + 1;

/// Kernel size for each layer in the 3-layer Conv1D architecture.
pub const KERNEL_SIZE: usize = 9;

/// Dilation factors for the 3 Conv1D layers (Conv-TasNet-inspired receptive field expansion).
/// Receptive field = 1 + (9-1)*(1 + 2 + 4) = 57 samples (> 32-sample chirp + noise context).
pub const DILATIONS: [usize; 3] = [1, 2, 4];

/// Maximum hidden channel width supported by static scratchpad buffers in `no_std`.
pub const MAX_HIDDEN_CHANNELS: usize = 32;

/// Resource accounting summary for comparing classical and neural filters on ESP32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceFootprint {
    pub weight_bytes: usize,
    pub metadata_bytes: usize,
    pub peak_activation_bytes: usize,
    pub macs_per_frame: usize,
}

impl ResourceFootprint {
    /// Total static model size (weights + quantization/bias metadata) in bytes.
    pub const fn model_bytes(&self) -> usize {
        self.weight_bytes + self.metadata_bytes
    }

    /// Total runtime SRAM footprint if weights + activations both reside in SRAM.
    pub const fn total_sram_bytes(&self) -> usize {
        self.model_bytes() + self.peak_activation_bytes
    }
}
