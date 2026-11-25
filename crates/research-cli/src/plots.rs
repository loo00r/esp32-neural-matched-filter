//! Publication-quality SVG figure generator (zero external plotting dependencies).

use std::fmt::Write as _;
use std::path::Path;

use neural_filter_core::FRAME_LEN;

/// One point on the Pareto trade-off frontier.
pub struct ParetoPlotPoint {
    pub label: String,
    pub macs_k: f64,
    pub si_snr_fp32: f32,
    pub si_snr_int8: f32,
    pub sram_int8_kb: f64,
}

/// One SNR sweep point.
pub struct SnrPlotPoint {
    pub snr_db: f32,
    pub cmf_awgn: f32,
    pub nmf_int8_awgn: f32,
    pub cmf_harsh: f32,
    pub nmf_fp32_harsh: f32,
    pub nmf_int8_harsh: f32,
}

/// Writes `results/pareto_tradeoff.svg`: SI-SNR vs MACs/frame (FP32 & INT8) vs Classical MF.
pub fn write_pareto_svg(
    path: &Path,
    classical_si_snr: f32,
    points: &[ParetoPlotPoint],
) -> std::io::Result<()> {
    let w = 820.0_f64;
    let h = 440.0_f64;
    let pad_l = 78.0;
    let pad_r = 40.0;
    let pad_t = 55.0;
    let pad_b = 70.0;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    let min_x = 1.0_f64.log10(); // 1k MACs
    let max_x = 2000.0_f64.log10(); // 2000k MACs
    let min_y = (classical_si_snr.min(-4.0) - 2.0) as f64;
    let max_y = points
        .iter()
        .map(|p| p.si_snr_fp32.max(p.si_snr_int8) as f64)
        .fold(14.0_f64, f64::max)
        + 2.0;

    let x_map = |macs_k: f64| -> f64 {
        let lx = macs_k.max(1.0).log10();
        pad_l + (lx - min_x) / (max_x - min_x) * plot_w
    };
    let y_map = |db: f64| -> f64 { pad_t + (1.0 - (db - min_y) / (max_y - min_y)) * plot_h };

    let mut svg = String::new();
    writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">"
    )
    .unwrap();
    writeln!(svg, "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>").unwrap();
    writeln!(
        svg,
        "<text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#111827\">Pareto Trade-off: SI-SNR vs Compute Cost (Industrial Harsh Channel, ESP32)</text>",
        w * 0.5
    )
    .unwrap();

    // Gridlines
    for tick_x in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0] {
        let x = x_map(tick_x);
        writeln!(svg, "<line x1=\"{x:.1}\" y1=\"{pad_t}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>", pad_t + plot_h).unwrap();
        writeln!(svg, "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#4b5563\">{tick_x:.0}k</text>", pad_t + plot_h + 18.0).unwrap();
    }
    let y_step = 4.0;
    let mut yv = (min_y / y_step).ceil() * y_step;
    while yv <= max_y {
        let y = y_map(yv);
        writeln!(svg, "<line x1=\"{pad_l}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>", pad_l + plot_w).unwrap();
        writeln!(svg, "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#4b5563\">{yv:.0} dB</text>", pad_l - 8.0, y + 4.0).unwrap();
        yv += y_step;
    }

    // Axes
    writeln!(svg, "<rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"none\" stroke=\"#374151\" stroke-width=\"1.2\"/>").unwrap();

    // Classical MF baseline horizontal reference line
    let y_cmf = y_map(classical_si_snr as f64);
    writeln!(svg, "<line x1=\"{pad_l}\" y1=\"{y_cmf:.1}\" x2=\"{:.1}\" y2=\"{y_cmf:.1}\" stroke=\"#dc2626\" stroke-width=\"1.8\" stroke-dasharray=\"6,4\"/>", pad_l + plot_w).unwrap();
    writeln!(svg, "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#dc2626\">Classical Sliding MF ({classical_si_snr:.2} dB, 3.1k MACs)</text>", pad_l + 8.0, y_cmf - 6.0).unwrap();

    // FP32 curve
    let mut fp_pts = String::new();
    let mut q_pts = String::new();
    for p in points {
        let x = x_map(p.macs_k);
        let y_f = y_map(p.si_snr_fp32 as f64);
        let y_q = y_map(p.si_snr_int8 as f64);
        write!(fp_pts, "{x:.1},{y_f:.1} ").unwrap();
        write!(q_pts, "{x:.1},{y_q:.1} ").unwrap();
    }
    writeln!(svg, "<polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2.2\" points=\"{fp_pts}\"/>").unwrap();
    writeln!(svg, "<polyline fill=\"none\" stroke=\"#059669\" stroke-width=\"2.2\" stroke-dasharray=\"4,2\" points=\"{q_pts}\"/>").unwrap();

    for p in points {
        let x = x_map(p.macs_k);
        let y_f = y_map(p.si_snr_fp32 as f64);
        let y_q = y_map(p.si_snr_int8 as f64);
        writeln!(svg, "<circle cx=\"{x:.1}\" cy=\"{y_f:.1}\" r=\"4.5\" fill=\"#2563eb\"/>").unwrap();
        writeln!(svg, "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"8\" height=\"8\" fill=\"#059669\"/>", x - 4.0, y_q - 4.0).unwrap();
        writeln!(
            svg,
            "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" font-weight=\"bold\" fill=\"#111827\">{} ({:.1} KB)</text>",
            y_f.min(y_q) - 10.0,
            p.label,
            p.sram_int8_kb
        )
        .unwrap();
    }

    // Axis labels & legend
    writeln!(svg, "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#111827\">Operations per Frame (log scale, kMACs/frame)</text>", pad_l + plot_w * 0.5, h - 18.0).unwrap();
    writeln!(svg, "<text x=\"22\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#111827\" transform=\"rotate(-90 22 {})\">Output SI-SNR (dB)</text>", pad_t + plot_h * 0.5, pad_t + plot_h * 0.5).unwrap();

    let lx = pad_l + plot_w - 220.0;
    let ly = pad_t + plot_h - 68.0;
    writeln!(svg, "<rect x=\"{lx}\" y=\"{ly}\" width=\"210\" height=\"58\" rx=\"4\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>").unwrap();
    writeln!(svg, "<circle cx=\"{}\" cy=\"{}\" r=\"4\" fill=\"#2563eb\"/>", lx + 14.0, ly + 18.0).unwrap();
    writeln!(svg, "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#111827\">Conv1D Neural MF (FP32)</text>", lx + 26.0, ly + 22.0).unwrap();
    writeln!(svg, "<rect x=\"{}\" y=\"{}\" width=\"8\" height=\"8\" fill=\"#059669\"/>", lx + 10.0, ly + 36.0).unwrap();
    writeln!(svg, "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#111827\">Conv1D Neural MF (Static INT8)</text>", lx + 26.0, ly + 44.0).unwrap();

    writeln!(svg, "</svg>").unwrap();
    std::fs::write(path, svg)
}

/// Writes `results/snr_sweep.svg`: Detection ROC-AUC (%) vs Input Pulse SNR across AWGN and Industrial Harsh.
pub fn write_snr_sweep_svg(path: &Path, points: &[SnrPlotPoint]) -> std::io::Result<()> {
    let w = 820.0_f64;
    let h = 440.0_f64;
    let pad_l = 74.0;
    let pad_r = 36.0;
    let pad_t = 55.0;
    let pad_b = 70.0;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    let min_x = points.first().map(|p| p.snr_db as f64).unwrap_or(-16.0);
    let max_x = points.last().map(|p| p.snr_db as f64).unwrap_or(2.0);
    let min_y = 50.0_f64;
    let max_y = 102.0_f64;

    let x_map = |snr: f64| pad_l + (snr - min_x) / (max_x - min_x).max(1.0) * plot_w;
    let y_map = |pct: f64| pad_t + (1.0 - (pct - min_y) / (max_y - min_y).max(1.0)) * plot_h;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">").unwrap();
    writeln!(svg, "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>").unwrap();
    writeln!(svg, "<text x=\"{}\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"#111827\">Signal Detection ROC-AUC vs Per-Sample Pulse SNR: AWGN vs Industrial Harsh</text>", w * 0.5).unwrap();

    for p in points {
        let x = x_map(p.snr_db as f64);
        writeln!(svg, "<line x1=\"{x:.1}\" y1=\"{pad_t}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#e5e7eb\"/>", pad_t + plot_h).unwrap();
        writeln!(svg, "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#4b5563\">{:+.0} dB</text>", pad_t + plot_h + 18.0, p.snr_db).unwrap();
    }
    let mut yv = 50.0;
    while yv <= 100.0 {
        let y = y_map(yv);
        writeln!(svg, "<line x1=\"{pad_l}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"#e5e7eb\"/>", pad_l + plot_w).unwrap();
        writeln!(svg, "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#4b5563\">{yv:.0}%</text>", pad_l - 8.0, y + 4.0).unwrap();
        yv += 10.0;
    }
    writeln!(svg, "<rect x=\"{pad_l}\" y=\"{pad_t}\" width=\"{plot_w}\" height=\"{plot_h}\" fill=\"none\" stroke=\"#374151\" stroke-width=\"1.2\"/>").unwrap();

    let draw_series = |svg: &mut String, color: &str, dash: &str, extract: &dyn Fn(&SnrPlotPoint) -> f32| {
        let mut pts = String::new();
        for p in points {
            let x = x_map(p.snr_db as f64);
            let y = y_map(extract(p) as f64);
            write!(pts, "{x:.1},{y:.1} ").unwrap();
        }
        writeln!(svg, "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2.2\" stroke-dasharray=\"{dash}\" points=\"{pts}\"/>").unwrap();
        for p in points {
            let x = x_map(p.snr_db as f64);
            let y = y_map(extract(p) as f64);
            writeln!(svg, "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"3.5\" fill=\"{color}\"/>").unwrap();
        }
    };

    draw_series(&mut svg, "#9ca3af", "6,4", &|p| p.cmf_awgn);
    draw_series(&mut svg, "#6366f1", "3,3", &|p| p.nmf_int8_awgn);
    draw_series(&mut svg, "#dc2626", "", &|p| p.cmf_harsh);
    draw_series(&mut svg, "#2563eb", "", &|p| p.nmf_fp32_harsh);
    draw_series(&mut svg, "#059669", "5,2", &|p| p.nmf_int8_harsh);

    writeln!(svg, "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#111827\">Per-Sample Pulse SNR (dB)</text>", pad_l + plot_w * 0.5, h - 18.0).unwrap();
    writeln!(svg, "<text x=\"22\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#111827\" transform=\"rotate(-90 22 {})\">Detection ROC-AUC (%)</text>", pad_t + plot_h * 0.5, pad_t + plot_h * 0.5).unwrap();

    let lx = pad_l + plot_w - 275.0;
    let ly = pad_t + plot_h - 114.0;
    writeln!(svg, "<rect x=\"{lx}\" y=\"{ly}\" width=\"265\" height=\"102\" rx=\"4\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>").unwrap();
    let legend = [
        ("#dc2626", "", "Classical MF — Industrial Harsh"),
        ("#2563eb", "", "Neural Conv1D FP32 — Industrial Harsh"),
        ("#059669", "5,2", "Neural Conv1D INT8 — Industrial Harsh"),
        ("#9ca3af", "6,4", "Classical MF — AWGN (Optimal)"),
        ("#6366f1", "3,3", "Neural Conv1D INT8 — AWGN"),
    ];
    for (i, (col, dash, label)) in legend.iter().enumerate() {
        let y = ly + 18.0 + (i as f64) * 18.0;
        writeln!(svg, "<line x1=\"{}\" y1=\"{y}\" x2=\"{}\" y2=\"{y}\" stroke=\"{col}\" stroke-width=\"2.2\" stroke-dasharray=\"{dash}\"/>", lx + 10.0, lx + 34.0).unwrap();
        writeln!(svg, "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#111827\">{label}</text>", lx + 42.0, y + 4.0).unwrap();
    }

    writeln!(svg, "</svg>").unwrap();
    std::fs::write(path, svg)
}

/// Input waveforms and SI-SNR metrics for the 4-panel time-domain demo plot.
pub struct WaveformDemo<'a> {
    pub snr_db: f32,
    pub clean: &'a [f32; FRAME_LEN],
    pub noisy: &'a [f32; FRAME_LEN],
    pub cmf_out: &'a [f32; FRAME_LEN],
    pub nmf_int8_out: &'a [f32; FRAME_LEN],
    pub cmf_sisnr: f32,
    pub nmf_sisnr: f32,
}

/// Writes `results/waveform_demo.svg`: 4-panel time-domain comparison on a harsh industrial frame.
pub fn write_waveform_demo_svg(path: &Path, demo: &WaveformDemo<'_>) -> std::io::Result<()> {
    let w = 840.0_f64;
    let h = 560.0_f64;
    let pad_l = 64.0;
    let pad_r = 28.0;
    let panel_h = 104.0;
    let gap = 24.0;
    let plot_w = w - pad_l - pad_r;

    let mut svg = String::new();
    writeln!(svg, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">").unwrap();
    writeln!(svg, "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>").unwrap();
    writeln!(svg, "<text x=\"{}\" y=\"26\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"15\" font-weight=\"bold\" fill=\"#111827\">Time-Domain Waveform Recovery (Industrial Harsh Channel, Input SNR = {:+.1} dB)</text>", w * 0.5, demo.snr_db).unwrap();

    let panels: [(&str, &[f32; FRAME_LEN], &str); 4] = [
        ("1. Clean LFM Chirp Target", demo.clean, "#111827"),
        ("2. Received Frame (Colored + Tonal + Impulses + Nonlinear Clipping)", demo.noisy, "#6b7280"),
        (&format!("3. Classical Linear Matched FIR Output (SI-SNR = {:+.2} dB)", demo.cmf_sisnr), demo.cmf_out, "#dc2626"),
        (&format!("4. Neural Conv1D Static INT8 Output (SI-SNR = {:+.2} dB)", demo.nmf_sisnr), demo.nmf_int8_out, "#059669"),
    ];

    for (idx, (title, sig, color)) in panels.iter().enumerate() {
        let top = 46.0 + (idx as f64) * (panel_h + gap);
        let max_abs = sig.iter().fold(1e-3_f32, |m, &v| m.max(v.abs())) as f64 * 1.15;

        writeln!(svg, "<text x=\"{pad_l}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"bold\" fill=\"#1f2937\">{title}</text>", top - 5.0).unwrap();
        writeln!(svg, "<rect x=\"{pad_l}\" y=\"{top:.1}\" width=\"{plot_w}\" height=\"{panel_h}\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>").unwrap();
        let mid_y = top + panel_h * 0.5;
        writeln!(svg, "<line x1=\"{pad_l}\" y1=\"{mid_y:.1}\" x2=\"{:.1}\" y2=\"{mid_y:.1}\" stroke=\"#e5e7eb\" stroke-dasharray=\"3,3\"/>", pad_l + plot_w).unwrap();

        let mut pts = String::new();
        for (t, &v) in sig.iter().enumerate() {
            let x = pad_l + (t as f64) / ((FRAME_LEN - 1) as f64) * plot_w;
            let y = mid_y - ((v as f64) / max_abs) * (panel_h * 0.45);
            write!(pts, "{x:.1},{y:.1} ").unwrap();
        }
        writeln!(svg, "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.8\" points=\"{pts}\"/>").unwrap();
    }

    writeln!(svg, "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#4b5563\">Sample index n (0 .. 127, 8 ms frame @ 16 kHz)</text>", pad_l + plot_w * 0.5, h - 10.0).unwrap();
    writeln!(svg, "</svg>").unwrap();
    std::fs::write(path, svg)
}
