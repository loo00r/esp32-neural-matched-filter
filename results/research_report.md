# Neural Matched Filtering for ESP32-WROOM-32: Experimental Report

## 1. Experimental Methodology
- **Signal Model**: Hann-windowed LFM chirp preamble (`TEMPLATE_LEN = 32`, sweep `0.08 -> 0.34`, zero-mean, unit L2 energy) embedded at random arrival lag `tau in [8, 88]` inside an `FRAME_LEN = 128` sample baseband frame (`8 ms @ 16 kHz`).
- **SNR Convention**: Standard per-sample pulse SNR (`SNR_pulse = 10 log10(P_pulse / P_noise)`), where the 32-sample chirp gives `+15.05 dB` theoretical matched-filter processing gain in AWGN. Evaluated in the low-SNR IoT operating band `[-8, +2] dB` (and swept across `[-14, +4] dB`).
- **Architecture**: Lightweight 3-layer dilated Conv1D (`kernel_size = 9`, dilations `[1, 2, 4]`, receptive field = `57` samples), trained purely in Rust with exact analytical backpropagation on negative **SI-SNR** loss (Luo & Mesgarani, 2019).
- **Static INT8 Quantization**: True integer-only inference (`i8` weights per-output-channel, `i8` signed I/O, `u8` post-ReLU hidden activations with `zp = 0`, `i32` accumulators, Q31 fixed-point requantization — zero floats in the inference path).
- **Target Platform**: ESP32-WROOM-32 (Xtensa LX6 @ 240 MHz, 520 KB SRAM / ~320 KB usable DRAM; active power model: 165 mW @ 240 MHz, 85.8 mW @ 80 MHz).

## 2. Study 1 — Channel Adaptation Across 4 Physical Regimes (Conv1D C=16, SNR_pulse in [-8, +2] dB)

| Channel Regime | Input SI-SNR (dB) | CMF FIR SI-SNR (dB) | CMF AUC | CMF Pd@1% | CMF ToA<=1 | Neural FP32 SI-SNR (dB) | Neural INT8 SI-SNR (dB) | INT8 AUC | INT8 Pd@1% | INT8 ToA<=1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AWGN (Baseline) | -9.47 | -10.52 | 0.886 | 18.7% | 81.3% | +13.73 | +11.54 | 0.907 | 27.0% | 79.3% |
| Colored + Narrowband | -9.26 | -10.52 | 0.914 | 31.5% | 83.8% | +18.63 | +15.00 | 0.931 | 40.3% | 81.3% |
| Nonlinear + Colored | -14.33 | -18.88 | 0.714 | 5.2% | 60.8% | +2.99 | +2.11 | 0.794 | 2.0% | 62.2% |
| Industrial Harsh (Non-AWGN) | -16.88 | -19.00 | 0.557 | 1.3% | 28.7% | +6.98 | +6.14 | 0.889 | 20.0% | 78.3% |

## 3. Study 2 — Accuracy vs Memory & Energy Pareto Trade-Off (Industrial Harsh Channel)

| Filter / Model | Precision | Params | SI-SNR (dB) | Quant Drop (dB) | ROC-AUC | Pd@1% | ToA<=1 | Weights (B) | Peak SRAM (KB) | % of 320KB DRAM | MACs/frame | Est. Latency @240MHz (us) | Est. Energy @240MHz (mJ) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical Dot-Product (sync) | FP32 | 128 | — | — | — | — | — | 512 | 1.00 | 0.31% | 128 | 7.9 | 0.00131 |
| Classical Sliding MF | FP32 | 32 | -19.56 | — | 0.532 | 2.3% | 24.0% | 128 | 1.00 | 0.31% | 3104 | 76.1 | 0.01256 |
| Classical Sliding MF | INT8 | 32 | -19.56 | ~0.00 | 0.532 | 2.3% | 24.0% | 32 | 0.54 | 0.17% | 3104 | 63.2 | 0.01043 |
| Conv1D Tiny-C4 | FP32 | 225 | -4.25 | — | 0.864 | 7.2% | 78.5% | 864 | 5.88 | 1.84% | 27648 | 638.6 | 0.10537 |
| Conv1D Tiny-C4 | **INT8** | 225 | **-4.29** | -0.04 | **0.862** | **6.7%** | **79.8%** | **216** (4.0x) | **2.07** | **0.65%** | 27648 | **523.4** | **0.08636** |
| Conv1D Small-C8 | FP32 | 737 | +6.75 | — | 0.839 | 14.0% | 73.0% | 2880 | 11.88 | 3.71% | 92160 | 2117.0 | 0.34930 |
| Conv1D Small-C8 | **INT8** | 737 | **+5.28** | -1.48 | **0.836** | **11.2%** | **72.0%** | **720** (4.0x) | **3.66** | **1.14%** | 92160 | **1733.0** | **0.28595** |
| Conv1D Medium-C16 | FP32 | 2625 | +9.94 | — | 0.881 | 28.2% | 78.2% | 10368 | 27.25 | 8.52% | 331776 | 7608.2 | 1.25535 |
| Conv1D Medium-C16 | **INT8** | 2625 | **+7.73** | -2.21 | **0.874** | **29.0%** | **75.8%** | **2592** (4.0x) | **7.68** | **2.40%** | 331776 | **6225.8** | **1.02726** |
| Conv1D Large-C32 | FP32 | 9857 | +11.75 | — | 0.894 | 33.8% | 80.3% | 39168 | 71.50 | 22.34% | 1253376 | 28728.2 | 4.74015 |
| Conv1D Large-C32 | **INT8** | 9857 | **+9.17** | -2.58 | **0.891** | **31.7%** | **79.3%** | **9792** (4.0x) | **19.08** | **5.96%** | 1253376 | **23505.8** | **3.87846** |

## 4. Study 3 — Per-Sample Pulse SNR Sweep (`-14 dB` to `+4 dB`, Medium-C16)

| Pulse SNR (dB) | AWGN CMF AUC | AWGN NMF-INT8 AUC | Harsh CMF AUC | Harsh CMF Pd@1% | Harsh NMF-FP32 AUC | Harsh NMF-INT8 AUC | Harsh NMF-INT8 Pd@1% | Harsh NMF-INT8 SI-SNR (dB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -14.0 | 0.534 | 0.505 | 0.497 | 1.1% | 0.537 | 0.538 | 2.0% | -18.86 |
| -11.0 | 0.592 | 0.576 | 0.526 | 3.1% | 0.599 | 0.599 | 2.6% | -15.52 |
| -8.0 | 0.703 | 0.677 | 0.535 | 3.4% | 0.667 | 0.665 | 9.1% | -7.68 |
| -5.0 | 0.831 | 0.802 | 0.578 | 4.6% | 0.839 | 0.839 | 30.0% | +1.42 |
| -2.0 | 0.972 | 0.958 | 0.645 | 1.1% | 0.936 | 0.933 | 55.7% | +11.33 |
| +1.0 | 0.999 | 0.995 | 0.684 | 5.7% | 0.991 | 0.982 | 86.9% | +20.64 |
| +4.0 | 1.000 | 0.999 | 0.814 | 20.3% | 0.997 | 0.964 | 92.9% | +23.33 |

## 5. Adaptive Classical / Neural Switching (DVS/DVFS Integration Prototype)

Tested on a 50/50 mixed stream of clean AWGN and Industrial Harsh frames (`SNR_pulse in [-8, +2] dB`):
- Clean AWGN frames routed to low-power Classical MF (80 MHz): **98.2%**
- Industrial Harsh frames routed to Neural INT8 MF (240 MHz): **100.0%**
- Mixed-stream Detection ROC-AUC: Classical-only `0.735` -> Adaptive `0.839` (Always-Neural `0.833`).
- Mixed-stream `Pd @ Pfa=1%`: Classical-only `24.4%` -> Adaptive `46.1%` (Always-Neural `31.2%`).
- Estimated Average Energy per Frame: Always-Neural `1027.3 uJ` -> Adaptive `531.5 uJ` (**48.3% energy reduction**).

## 6. Generated Figures
- `pareto_tradeoff.svg`: SI-SNR vs Compute/SRAM Pareto frontier
- `snr_sweep.svg`: Detection ROC-AUC (%) vs Input Pulse SNR across AWGN and Industrial Harsh channels
- `waveform_demo.svg`: 4-panel time-domain recovery comparison on Industrial Harsh frame at `-3 dB`

## 7. Compiled ESP32-WROOM-32 Firmware Memory Profile (`xtensa-esp32-elf-size`)

Cross-compiled with `cargo +esp build --release` (`target = xtensa-esp32-none-elf`, `opt-level = 3`, `lto = "fat"`), sharing the exact `no_std` `neural-filter-core` crate and Pareto-optimal `Medium-C16` weights:

| ELF Section / Metric | Production INT8-Only (`--no-default-features`) | Full A/B Benchmark (`INT8 + FP32`) | Difference (FP32 Overhead) |
|---|---:|---:|---:|
| Static DRAM `.bss` | **8,360 B (8.16 KB)** | 41,132 B (40.17 KB) | +32,772 B (+32.0 KB) |
| Static DRAM `.data` | **1,276 B (1.25 KB)** | 1,280 B (1.25 KB) | +4 B |
| **Total Static DRAM (`.data + .bss`)** | **9,636 B (9.41 KB — 3.0% of 320 KB DRAM / 1.8% of 520 KB SRAM)** | 42,412 B (41.42 KB — 13.0% of DRAM) | **4.40x less static DRAM** |
| Free Stack/Heap DRAM (`.stack`) | **186,972 B (182.6 KB free)** | 154,196 B (150.6 KB free) | +32.0 KB saved |
| Flash `.rodata` (const weights & tables) | **12,632 B (12.34 KB)** | 23,504 B (22.95 KB) | +10,872 B (+10.62 KB) |
| Flash `.text` (compiled machine code) | **42,149 B (41.16 KB)** | 44,365 B (43.33 KB) | +2,216 B |
| IRAM (`.rwtext + .vectors`) | **5,848 B (5.71 KB)** | 5,848 B (5.71 KB) | 0 B |
| **Total Flash Image** | **62,161 B (~60.7 KB — 1.5% of 4 MB Flash)** | 75,253 B (~73.5 KB) | **-13.1 KB Flash** |

> **Verification**: Static INT8 weights for `Medium-C16` occupy `2,592 B` vs `10,368 B` for FP32 (**exact 4.00x reduction**). Static `no_std` INT8 inference uses zero heap allocation and zero floating-point instructions inside the Conv1D kernels.

## 8. Real Hardware Benchmarks on Physical ESP32-WROOM-32 (rev. v3.1, 240 MHz)

Measured directly on physical **ESP32-WROOM-32 (rev. v3.1, 40 MHz crystal, 4 MB Flash, CPU = 240 MHz)** using Xtensa hardware cycle counter `CCOUNT` averaged over 100 iterations after cache warm-up:

- **On-Chip Bit-Exact Parity**: `INT8 Checksum = 0x72EB37E5` (matches host `0x72EB37E5` — **BIT-EXACT PASS**).
- **On-Chip Time-of-Arrival Check (`industrial_harsh`, true `lag = 32`)**: Classical MF failed (`lag = 95`, error = `63` samples); Neural Static INT8 recovered **`lag = 32` (0-sample error)**.

| Kernel (ESP32-WROOM-32 @ 240 MHz) | Measured Cycles (`CCOUNT`) | Measured Latency (`us`) | Est. Energy @ 165 mW (`uJ`) | Est. Energy (`mJ/frame`) | Cycles / MAC |
|---|---:|---:|---:|---:|---:|
| Classical Dot-Product (`f32`, 128 MACs) | 1,616 | 6.7 | 1.11 | 0.00111 | 12.6 |
| Adaptive Mode Diagnostics (`O(L)`) | 4,329 | 18.0 | 2.98 | 0.00298 | — |
| Classical Sliding MF (`f32`, 3,104 MACs) | 17,317 | 72.2 | 11.91 | 0.01191 | 5.58 |
| Classical Sliding MF (`i8`, 3,104 MACs) | 38,045 | 158.5 | 26.16 | 0.02616 | 12.26 |
| Neural Conv1D `Medium-C16` (`FP32`, 331,776 MACs) | 4,691,533 | 19,548.1 | 3,225.43 | 3.22543 | 14.14 |
| Neural Conv1D `Medium-C16` (`Static INT8`, 331,776 MACs) | 5,653,652 | 23,556.9 | 3,886.89 | 3.88689 | 17.04 |

**Key Microarchitectural Finding (Xtensa LX6)**:
- On ESP32-WROOM-32 (Xtensa LX6), scalar `FP32` inference (`19.55 ms`) executes **1.20x faster** than scalar `INT8` (`23.56 ms`, speedup `0.83x`) because Xtensa LX6 features a hardware single-precision FPU with fused `MADD.S`, whereas it lacks dedicated 128-bit INT8 SIMD instructions (introduced later in ESP32-S3 PIE). Thus, on ESP32-WROOM-32, **static INT8 quantization's decisive advantage is 4.00x weight compression (`10.37 KB -> 2.59 KB`) and 4.40x static DRAM savings (`41.42 KB -> 9.41 KB`)**, enabling deployment well within strict SRAM budgets.
- Using real measured `CCOUNT` energies, adaptive switching (`98.2%` AWGN -> Classical `f32`, `100%` Industrial Harsh -> Neural INT8, `18.0 us` diagnostic overhead) lowers average frame energy on a 50/50 mixed stream from `3.887 mJ` to **`1.987 mJ` (`48.9%` saved)**, and at 20% harsh-burst duty cycle to **`0.790 mJ` (`79.7%` saved)**.


