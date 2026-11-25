#![forbid(unsafe_code)]

//! Pure-Rust CLI driving training, static INT8 quantization, Pareto evaluation,
//! SVG plotting, and ESP32-WROOM-32 static weight export.

mod experiment;
mod export;
mod metrics;
mod plots;
mod quantize;
mod rng;
mod signal;
mod train;

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let quick = args.iter().any(|a| a == "--quick");

    let results_dir = PathBuf::from("results");
    let firmware_export = PathBuf::from("firmware/esp32-benchmark/src/generated_model.rs");

    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") => {
            println!("Usage: research-cli [--quick]");
            println!("  Runs the complete Neural Matched Filter research suite:");
            println!("  - Study 1: 4-regime channel adaptation (AWGN, Colored, Nonlinear+Colored, Industrial Harsh)");
            println!("  - Study 2: 3-layer Conv1D Pareto sweep (C = 4, 8, 16, 32) FP32 vs Static INT8");
            println!("  - Study 3: SNR sweep (-10..+10 dB) & Adaptive Mode Controller");
            println!("  - Exports ESP32 firmware weights to firmware/esp32-benchmark/src/generated_model.rs");
            println!("  - Writes CSVs, Markdown report, and SVG plots to results/");
        }
        _ => {
            experiment::run_full_suite(&results_dir, &firmware_export, quick)?;
        }
    }

    Ok(())
}
