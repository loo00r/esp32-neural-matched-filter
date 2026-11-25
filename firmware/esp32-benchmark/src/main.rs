#![no_std]
#![no_main]

//! Bare-metal ESP32-WROOM-32 (Xtensa LX6 @ 240 MHz) hardware benchmark for:
//! 1. Classical Matched Filter — synchronized dot product (`f32`) & sliding correlation (`f32` / `i8`)
//! 2. 3-Layer Dilated Conv1D Neural Matched Filter — Static INT8 (zero-float integer-only)
//! 3. 3-Layer Dilated Conv1D Neural Matched Filter — FP32 (when `bench-fp32` feature is enabled)
//! 4. Adaptive Classical <-> Neural Mode Controller (`O(L)` frame diagnostics)
//!
//! Uses Xtensa `CCOUNT` (`xtensa_lx::timer::get_cycle_count`) for cycle-exact timing and verifies
//! bit-exact parity of INT8 inference against the host-computed FNV-1a checksum.

mod generated_model;

use core::cell::UnsafeCell;
use core::hint::black_box;
use core::sync::atomic::{AtomicBool, Ordering};

use esp_hal::clock::CpuClock;
use esp_println::println;
use neural_filter_core::{
    FRAME_LEN, VALID_CORR_LEN,
    adaptive::{AdaptiveThresholds, select_filter_mode},
    classical::{
        dot_product_f32, sliding_matched_filter_f32, sliding_matched_filter_i8,
    },
    conv1d_int8::ScratchpadInt8,
    quant::dequantize_frame_i8,
};

#[cfg(feature = "bench-fp32")]
use neural_filter_core::conv1d_f32::ScratchpadF32;

esp_bootloader_esp_idf::esp_app_desc!();

/// Single-owner static BSS container so `xtensa-esp32-elf-size` captures exact SRAM usage
/// without relying on heap allocation or large stack frames.
struct StaticBuffer<T> {
    taken: AtomicBool,
    inner: UnsafeCell<T>,
}

unsafe impl<T> Sync for StaticBuffer<T> {}

impl<T> StaticBuffer<T> {
    const fn new(val: T) -> Self {
        Self {
            taken: AtomicBool::new(false),
            inner: UnsafeCell::new(val),
        }
    }

    fn take(&self) -> &'static mut T {
        assert!(!self.taken.swap(true, Ordering::AcqRel));
        // SAFETY: `taken` guarantees unique mutable access for the lifetime of `main`.
        unsafe { &mut *self.inner.get() }
    }
}

static SCRATCH_INT8: StaticBuffer<ScratchpadInt8> = StaticBuffer::new(ScratchpadInt8::new());

#[cfg(feature = "bench-fp32")]
static SCRATCH_FP32: StaticBuffer<ScratchpadF32> = StaticBuffer::new(ScratchpadF32::new());

const BENCH_ITERS: u32 = 100;
const CPU_HZ: u64 = 240_000_000;
// Datasheet reference active power @ 240 MHz, 3.3V, single core, Wi-Fi/BT off (~50 mA => 165 mW)
const POWER_MW_240MHZ: f64 = 165.0;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
    println!("[PANIC] {}", info);
    loop {
        core::hint::spin_loop();
    }
}

fn fnv1a_i8(data: &[i8]) -> u32 {
    let mut h = 0x811C_9DC5_u32;
    for &b in data {
        h ^= (b as u8) as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn measure_cycles<F: FnMut()>(iters: u32, mut f: F) -> u32 {
    // Warm-up pass (fills instruction cache & branch predictor)
    f();
    let mut total: u64 = 0;
    for _ in 0..iters {
        let start = xtensa_lx::timer::get_cycle_count();
        f();
        let end = xtensa_lx::timer::get_cycle_count();
        total += end.wrapping_sub(start) as u64;
    }
    (total / iters as u64) as u32
}

fn print_bench_row(name: &str, avg_cycles: u32) {
    let latency_us = (avg_cycles as f64) * 1e6 / (CPU_HZ as f64);
    let energy_uj = POWER_MW_240MHZ * latency_us * 1e-3;
    let energy_mj = energy_uj * 1e-3;
    println!(
        "| {:<30} | {:>9} cyc | {:>8.1} us | {:>7.2} uJ ({:.5} mJ) |",
        name, avg_cycles, latency_us, energy_uj, energy_mj
    );
}

#[esp_hal::main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    println!("\n======================================================================");
    println!(" ESP32-WROOM-32 Neural Matched Filter Hardware Benchmark (240 MHz)");
    println!(" Model: {} (C={})", generated_model::MODEL_NAME, generated_model::HIDDEN_CHANNELS);
    println!(" Frame: {} samples | Template: {} samples | Iters: {}", FRAME_LEN, neural_filter_core::TEMPLATE_LEN, BENCH_ITERS);
    println!("======================================================================");

    let scratch_q = SCRATCH_INT8.take();
    let mut out_q = [0_i8; FRAME_LEN];
    let mut out_deq = [0.0_f32; FRAME_LEN];
    let mut corr_f32 = [0.0_f32; VALID_CORR_LEN];
    let mut corr_i32 = [0_i32; VALID_CORR_LEN];

    // 1. Bit-exact host parity verification on embedded IndustrialHarsh test frame
    generated_model::MODEL_INT8.infer_quantized(
        &generated_model::TEST_FRAME_I8,
        &mut out_q,
        scratch_q,
    );
    let checksum = fnv1a_i8(&out_q);
    let parity_ok = checksum == generated_model::REF_INT8_OUT_CHECKSUM;
    dequantize_frame_i8(&out_q, generated_model::OUTPUT_SCALE, &mut out_deq);
    let nmf_peak = sliding_matched_filter_f32(&out_deq, &generated_model::TEMPLATE_F32, &mut corr_f32);
    let cmf_peak = sliding_matched_filter_f32(
        &generated_model::TEST_FRAME_F32,
        &generated_model::TEMPLATE_F32,
        &mut corr_f32,
    );
    println!(
        "Verification [{}]: INT8 Checksum 0x{:08X} (expected 0x{:08X}) => {}",
        generated_model::TEST_FRAME_LABEL,
        checksum,
        generated_model::REF_INT8_OUT_CHECKSUM,
        if parity_ok { "BIT-EXACT PASS" } else { "MISMATCH!" }
    );
    println!(
        "ToA Check (true lag={}): Classical MF lag={} | Neural INT8 lag={}\n",
        generated_model::TEST_FRAME_TRUE_LAG,
        cmf_peak.peak_index,
        nmf_peak.peak_index
    );

    // 2. Cycle-exact CCOUNT hardware benchmarks
    println!("| Kernel                         |    Cycles |   Latency |        Est. Energy @240MHz |");
    println!("|--------------------------------|-----------|-----------|----------------------------|");

    let cyc_dot = measure_cycles(BENCH_ITERS, || {
        let v = dot_product_f32(
            black_box(&generated_model::TEST_FRAME_F32),
            black_box(&generated_model::TEST_FRAME_F32),
        );
        black_box(v);
    });
    print_bench_row("Classical Dot-Product (f32)", cyc_dot);

    let cyc_smf_f32 = measure_cycles(BENCH_ITERS, || {
        let p = sliding_matched_filter_f32(
            black_box(&generated_model::TEST_FRAME_F32),
            black_box(&generated_model::TEMPLATE_F32),
            &mut corr_f32,
        );
        black_box(p);
    });
    print_bench_row("Classical Sliding MF (f32)", cyc_smf_f32);

    let cyc_smf_i8 = measure_cycles(BENCH_ITERS, || {
        let p = sliding_matched_filter_i8(
            black_box(&generated_model::TEST_FRAME_I8),
            black_box(&generated_model::TEMPLATE_I8),
            &mut corr_i32,
        );
        black_box(p);
    });
    print_bench_row("Classical Sliding MF (i8)", cyc_smf_i8);

    let cyc_adapt = measure_cycles(BENCH_ITERS, || {
        let res = select_filter_mode(
            black_box(&generated_model::TEST_FRAME_F32),
            AdaptiveThresholds::default(),
        );
        black_box(res);
    });
    print_bench_row("Adaptive Mode Diagnostics", cyc_adapt);

    let cyc_nmf_i8 = measure_cycles(BENCH_ITERS, || {
        generated_model::MODEL_INT8.infer_quantized(
            black_box(&generated_model::TEST_FRAME_I8),
            &mut out_q,
            scratch_q,
        );
        black_box(&out_q);
    });
    print_bench_row("Neural Conv1D Static INT8", cyc_nmf_i8);

    #[cfg(feature = "bench-fp32")]
    {
        let scratch_f = SCRATCH_FP32.take();
        let mut out_f = [0.0_f32; FRAME_LEN];
        let cyc_nmf_f32 = measure_cycles(BENCH_ITERS, || {
            generated_model::MODEL_F32.infer(
                black_box(&generated_model::TEST_FRAME_F32),
                &mut out_f,
                scratch_f,
            );
            black_box(&out_f);
        });
        print_bench_row("Neural Conv1D FP32", cyc_nmf_f32);
        println!(
            "\nINT8 vs FP32 Speedup: {:.2}x | INT8 vs Classical Sliding MF (i8): {:.1}x cycles",
            (cyc_nmf_f32 as f64) / (cyc_nmf_i8.max(1) as f64),
            (cyc_nmf_i8 as f64) / (cyc_smf_i8.max(1) as f64)
        );
    }

    println!("Benchmark complete. Idling...");
    loop {
        core::hint::spin_loop();
    }
}
