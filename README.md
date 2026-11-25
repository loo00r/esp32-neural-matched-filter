# Neural Matched Filtering for Energy-Efficient ESP32-WROOM-32 IoT Nodes

Pure-Rust research framework, training/quantization pipeline, and bare-metal `no_std` ESP32-WROOM-32 firmware implementing a **3-layer dilated Conv1D Neural Matched Filter** trained with **SI-SNR** loss and deployed via **integer-only static INT8 quantization**.

---

## Architecture & Clean-Code Layout

```text
.
├── crates/
│   ├── neural-filter-core/        # #![no_std], #![forbid(unsafe_code)], zero-alloc, zero deps
│   │   └── src/
│   │       ├── classical.rs       # Classical Matched Filter (sync dot-product & sliding FIR, f32/i8)
│   │       ├── conv1d_f32.rs      # 3-layer dilated Conv1D FP32 kernel (dilations [1, 2, 4], K=9)
│   │       ├── conv1d_int8.rs     # Pure-integer static INT8 Conv1D kernel (i8 weights, u8 ReLU, i32 acc)
│   │       ├── quant.rs           # Fixed-point Q31 requantization (zero floats in inference)
│   │       └── adaptive.rs        # O(L) non-AWGN detector for Classical <-> Neural DVS/DVFS switching
│   │
│   └── research-cli/              # Pure-Rust host training, static PTQ, evaluation & codegen (zero deps)
│       └── src/
│           ├── signal.rs          # LFM chirp preamble + 4 channel regimes (AWGN..IndustrialHarsh)
│           ├── metrics.rs         # SI-SNR + exact analytical grad, ROC-AUC, Pd@Pfa, ESP32 energy model
│           ├── train.rs           # Multi-threaded analytical backprop + AdamW + cosine LR schedule
│           ├── quantize.rs        # Static PTQ: per-channel symmetric weights + percentile calibration
│           ├── export.rs          # Generates static const Rust tables for ESP32 firmware
│           ├── plots.rs           # Zero-dependency publication SVG generator
│           └── experiment.rs      # Complete 3-study reproducible research suite
│
├── firmware/
│   └── esp32-benchmark/           # Standalone xtensa-esp32-none-elf bare-metal firmware (esp-hal 1.0)
│       └── src/
│           ├── main.rs            # Hardware CCOUNT cycle benchmark + bit-exact host parity check
│           └── generated_model.rs # Auto-generated static Medium-C16 INT8 & FP32 weights + test vector
│
└── results/
    ├── research_report.md         # Full tables, Pareto frontier, SNR sweep & compiled ELF sizes
    ├── regimes.csv                # Study 1 raw metrics
    ├── pareto.csv                 # Study 2 raw metrics
    ├── snr_sweep.csv              # Study 3 raw metrics
    ├── pareto_tradeoff.svg        # Accuracy vs MACs/SRAM Pareto plot
    ├── snr_sweep.svg              # ROC-AUC vs Per-Sample Pulse SNR plot
    └── waveform_demo.svg          # 4-panel time-domain recovery at -3 dB Industrial Harsh
```

---

## Key Experimental Findings

- **Channel Adaptation (`Conv1D C=16`, `SNR_pulse in [-8, +2] dB`)**:
  - Under **AWGN**, Classical Matched Filter achieves near-optimal detection (`AUC = 0.886`, `ToA <= 1 smp: 81.3%`).
  - Under **Industrial Harsh** interference (colored AR(1) + narrowband hum + heavy-tailed switching impulses + nonlinear saturation + multipath), Classical MF collapses (`AUC = 0.557`, `Pd@1% = 1.3%`, `ToA <= 1: 28.7%`), while **Static INT8 Conv1D** restores **`AUC = 0.889`**, **`Pd@1% = 20.0%`** (**15x higher**), **`ToA <= 1: 78.3%`**, and **`SI-SNR = +6.14 dB`**.
- **Pareto Sweet Spot on ESP32-WROOM-32 (`Medium-C16`)**:
  - `2,625` parameters; **exact 4.00x weight compression** (`10,368 B` FP32 -> `2,592 B` INT8).
  - Real compiled `xtensa-esp32-none-elf` production INT8-only firmware occupies **`9.41 KB` total static DRAM** (`3.0%` of 320 KB usable DRAM / `1.8%` of 520 KB SRAM), leaving `182.6 KB` free stack/heap, and `60.7 KB` Flash (`1.5%` of 4 MB Flash).
- **Adaptive Classical / Neural Switching (DVS/DVFS Prototype)**:
  - Routes `98.2%` of clean AWGN frames to low-power Classical MF (80 MHz) and `100.0%` of Industrial Harsh frames to Neural INT8 (240 MHz), improving mixed-stream `Pd@1%` from `24.4%` (classical) / `31.2%` (always-neural) to **`46.1%`** while saving **`48.3%` average energy per frame**.

---

## Quick Start

### 1. Run Unit Tests (including analytical vs finite-difference gradient verification)
```bash
cargo test --workspace
```

### 2. Reproduce All Experiments, Plots, CSVs, and Regenerate ESP32 Weights
```bash
cargo run --release -p research-cli
# Faster smoke run (~15s):
cargo run --release -p research-cli -- --quick
```

### 3. Cross-Compile ESP32-WROOM-32 Firmware (no board required)
```bash
cd firmware/esp32-benchmark

# Production INT8-only build (minimal SRAM/Flash footprint)
cargo +esp build --release --no-default-features

# Full A/B benchmark build (INT8 vs FP32 vs Classical MF)
cargo +esp build --release

# Inspect exact ELF memory sections
xtensa-esp32-elf-size -A target/xtensa-esp32-none-elf/release/esp32-benchmark
```

### 4. Flash & Collect Real Hardware `CCOUNT` Benchmarks (when ESP32-WROOM-32 is connected via USB)
```bash
cd firmware/esp32-benchmark
cargo +esp run --release
```
The firmware verifies bit-exact INT8 output parity against the host FNV-1a checksum (`REF_INT8_OUT_CHECKSUM`) and prints averaged `CCOUNT` cycles, latency (`us`), and estimated energy (`uJ`/`mJ`) at 240 MHz over UART.
