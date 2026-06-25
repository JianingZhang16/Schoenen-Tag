#!/usr/bin/env python3
"""DATABASE PPG FFT and ACC-NLMS FFT baselines against BPM0.

This script reads the local TROIKA-style DATABASE folder and compares:

1. PPG-only FFT baseline.
2. ACC-referenced NLMS artifact cancellation followed by FFT HR estimation.

BPM0 is used as the ECG-derived window-level ground truth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg_fft_hr_analysis import (
    DEFAULT_DATA_DIR,
    DatabaseRecord,
    bandpass_signal,
    build_metrics,
    discover_records,
    estimate_ppg_fft_record,
    extract_acc_channels,
    extract_ppg_channels,
    fft_peak_hr,
    load_record,
    postprocess_by_record,
    zscore,
)


DEFAULT_OUTDIR = Path("outputs_database_ppg_nlms_acc_fft_hr")


def nlms_artifact_cancel(
    desired: np.ndarray,
    references: np.ndarray,
    filter_order: int,
    mu: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use multi-reference NLMS to estimate ACC-correlated artifact and return the error signal."""
    desired = np.asarray(desired, dtype=float)
    references = np.asarray(references, dtype=float)
    if references.ndim != 2:
        raise ValueError("references must be samples x channels")
    if desired.shape[0] != references.shape[0]:
        raise ValueError("desired and references must have the same sample count")
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1")
    if not (0.0 < mu < 2.0):
        raise ValueError("NLMS mu should be in (0, 2)")

    d = zscore(desired)
    x = np.column_stack([zscore(references[:, i]) for i in range(references.shape[1])])
    n_samples, n_channels = x.shape
    weights = np.zeros(filter_order * n_channels, dtype=float)
    artifact = np.zeros(n_samples, dtype=float)
    cleaned = np.zeros(n_samples, dtype=float)

    for n in range(n_samples):
        x_vec = np.zeros(filter_order * n_channels, dtype=float)
        available = min(filter_order, n + 1)
        history = x[n - available + 1 : n + 1][::-1]
        x_vec[: available * n_channels] = history.reshape(-1)

        y_hat = float(weights @ x_vec)
        error = d[n] - y_hat
        weights += (mu / (float(x_vec @ x_vec) + eps)) * error * x_vec

        artifact[n] = y_hat
        cleaned[n] = error

    return zscore(cleaned), artifact


def make_nlms_cleaned_ppg(
    record: DatabaseRecord,
    sig: np.ndarray,
    fs: float,
    ppg_low_hz: float,
    ppg_high_hz: float,
    filter_order: int,
    nlms_filter_order: int,
    nlms_mu: float,
    nlms_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ppg1, ppg2 = extract_ppg_channels(record, sig)
    acc = extract_acc_channels(record, sig)

    ppg1_proc = zscore(bandpass_signal(ppg1, fs=fs, low_hz=ppg_low_hz, high_hz=ppg_high_hz, order=filter_order))
    ppg2_proc = zscore(bandpass_signal(ppg2, fs=fs, low_hz=ppg_low_hz, high_hz=ppg_high_hz, order=filter_order))
    desired = 0.5 * ppg1_proc + 0.5 * ppg2_proc

    refs = []
    for axis in acc:
        refs.append(zscore(bandpass_signal(axis, fs=fs, low_hz=ppg_low_hz, high_hz=ppg_high_hz, order=filter_order)))
    references = np.column_stack(refs)

    cleaned, artifact = nlms_artifact_cancel(
        desired,
        references,
        filter_order=nlms_filter_order,
        mu=nlms_mu,
        eps=nlms_eps,
    )
    return desired, cleaned, artifact


def estimate_signal_fft_record(
    x: np.ndarray,
    bpm0: np.ndarray,
    record: DatabaseRecord,
    fs: float,
    window_sec: float,
    step_sec: float,
    n_fft: int,
    search_low_hz: float,
    search_high_hz: float,
    prefix: str,
) -> pd.DataFrame:
    window = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    window_nb = min(int(np.floor((len(x) - window) / step) + 1), len(bpm0))
    rows = []
    for idx in range(window_nb):
        cur = slice(idx * step, idx * step + window)
        segment = x[cur]
        peak_hz, hr_raw, peak_power = fft_peak_hr(
            segment,
            fs=fs,
            n_fft=n_fft,
            low_hz=search_low_hz,
            high_hz=search_high_hz,
        )
        start_s = idx * step_sec
        rows.append(
            {
                "record": record.name,
                "split": record.split,
                "fs_hz": fs,
                "window_index": idx + 1,
                "start_time_s": start_s,
                "end_time_s": start_s + window_sec,
                "center_time_s": start_s + window_sec / 2.0,
                "hr_true_bpm": float(bpm0[idx]),
                "ground_truth_hr_bpm": float(bpm0[idx]),
                f"{prefix}_peak_hz_raw": peak_hz,
                f"{prefix}_hr_raw_bpm": hr_raw,
                f"{prefix}_peak_power_raw": peak_power,
            }
        )
    return pd.DataFrame(rows)


def merge_on_window(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["record", "split", "fs_hz", "window_index", "start_time_s", "end_time_s", "center_time_s", "hr_true_bpm", "ground_truth_hr_bpm"]
    extra = [col for col in right.columns if col not in keys]
    return left.merge(right[keys + extra], on=keys, how="inner")


def add_errors(df: pd.DataFrame, methods: list[tuple[str, str]]) -> pd.DataFrame:
    df = df.copy()
    for label, col in methods:
        slug = label.lower().replace("-", "_").replace(" ", "_")
        if col not in df:
            continue
        df[f"{slug}_error_bpm"] = df[col] - df["hr_true_bpm"]
        df[f"{slug}_abs_error_bpm"] = np.abs(df[f"{slug}_error_bpm"])
        df[f"{slug}_relative_abs_error_percent"] = df[f"{slug}_abs_error_bpm"] / df["hr_true_bpm"] * 100.0
    return df


def plot_time_series(df: pd.DataFrame, record_name: str, outdir: Path) -> Path:
    record_df = df[df["record"] == record_name]
    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    ax.plot(record_df["center_time_s"] / 60.0, record_df["hr_true_bpm"], label="BPM0 true HR", linewidth=1.7, color="#1f77b4")
    ax.plot(record_df["center_time_s"] / 60.0, record_df["ppg_fft_hr_est_bpm"], label="PPG FFT", linewidth=1.05, color="#ff7f0e")
    ax.plot(record_df["center_time_s"] / 60.0, record_df["ppg_acc_nlms_fft_hr_est_bpm"], label="ACC-NLMS FFT", linewidth=1.2, color="#d62728")
    ax.set_title(f"DATABASE PPG FFT and ACC-NLMS FFT vs BPM0: {record_name}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / f"{record_name}_timeseries_ppg_fft_nlms_vs_bpm0.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mae(metrics: pd.DataFrame, outdir: Path) -> Path:
    records = ["ALL_training", "ALL_competition", "ALL_DATABASE"]
    methods = ["PPG FFT", "ACC-NLMS FFT"]
    colors = {"PPG FFT": "#ff7f0e", "ACC-NLMS FFT": "#d62728"}
    x = np.arange(len(records), dtype=float)
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for i, method in enumerate(methods):
        values = []
        for record in records:
            row = metrics[(metrics["record"] == record) & (metrics["method"] == method)]
            values.append(float(row["mae_bpm"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - 0.5) * width, values, width=width, label=method, color=colors[method])
    ax.set_xticks(x)
    ax.set_xticklabels(["Training", "Competition", "All"])
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("DATABASE MAE: PPG FFT vs ACC-NLMS FFT")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "mae_ppg_fft_vs_acc_nlms_fft.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    specs = [
        ("PPG FFT", "ppg_fft_hr_est_bpm", "#ff7f0e"),
        ("ACC-NLMS FFT", "ppg_acc_nlms_fft_hr_est_bpm", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.9), constrained_layout=True)
    vals = [df["hr_true_bpm"].to_numpy()]
    text_lines = []
    for label, col, color in specs:
        valid = df[["hr_true_bpm", col]].dropna()
        ax.scatter(valid["hr_true_bpm"], valid[col], s=12, alpha=0.32, label=label, color=color)
        vals.append(valid[col].to_numpy())
        row = metrics[(metrics["record"] == "ALL_DATABASE") & (metrics["method"] == label)]
        if not row.empty:
            row = row.iloc[0]
            text_lines.append(f"{label}: r={row['pearson_r']:.3f}, MAE={row['mae_bpm']:.2f}")
    finite = np.concatenate([v[np.isfinite(v)] for v in vals if v.size])
    lo, hi = float(finite.min() - 5), float(finite.max() + 5)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(0.04, 0.96, "\n".join(text_lines), transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("DATABASE HR correlation")
    ax.set_xlabel("BPM0 true HR (bpm)")
    ax.set_ylabel("Estimated HR (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    path = outdir / "scatter_ppg_fft_vs_acc_nlms_fft.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_bland_altman(df: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    specs = [
        ("PPG FFT", "ppg_fft_hr_est_bpm", "#ff7f0e"),
        ("ACC-NLMS FFT", "ppg_acc_nlms_fft_hr_est_bpm", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, (label, col, color) in zip(axes, specs):
        valid = df[["hr_true_bpm", col]].dropna()
        mean_hr = valid.mean(axis=1)
        diff_hr = valid[col] - valid["hr_true_bpm"]
        ax.scatter(mean_hr, diff_hr, s=12, alpha=0.35, color=color)
        row = metrics[(metrics["record"] == "ALL_DATABASE") & (metrics["method"] == label)]
        if not row.empty:
            row = row.iloc[0]
            bias = float(row["bias_bpm"])
            lower = float(row["bland_altman_lower_bpm"])
            upper = float(row["bland_altman_upper_bpm"])
            ax.axhline(bias, color=color, linewidth=1.2, label="Bias")
            ax.axhline(lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
            ax.axhline(upper, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Mean HR (bpm)")
        ax.set_ylabel("HR est - BPM0 (bpm)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    path = outdir / "bland_altman_ppg_fft_vs_acc_nlms_fft.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPG FFT and ACC-NLMS FFT HR estimation on DATABASE.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--records", nargs="+", default=["all"])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fs", type=float, default=125.0)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--search-low-hz", type=float, default=0.8)
    parser.add_argument("--search-high-hz", type=float, default=3.0)
    parser.add_argument("--ppg-low-hz", type=float, default=0.4)
    parser.add_argument("--ppg-high-hz", type=float, default=4.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--nlms-filter-order", type=int, default=32)
    parser.add_argument("--nlms-mu", type=float, default=0.005)
    parser.add_argument("--nlms-eps", type=float, default=1e-6)
    parser.add_argument("--physio-low-bpm", type=float, default=40.0)
    parser.add_argument("--physio-high-bpm", type=float, default=220.0)
    parser.add_argument("--smooth-windows", type=int, default=5)
    parser.add_argument("--outlier-policy", choices=["interpolate", "previous", "nan"], default="interpolate")
    parser.add_argument("--plot-records", nargs="+", default=["DATA_01_TYPE01", "DATA_08_TYPE02", "TEST_S01_T01", "TEST_S07_T02"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.data_dir, args.records)
    frames = []
    clean_summaries = []
    for record in records:
        sig, bpm0 = load_record(record)
        ppg_fft = estimate_ppg_fft_record(
            record,
            sig,
            bpm0,
            fs=args.fs,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            n_fft=args.n_fft,
            search_low_hz=args.search_low_hz,
            search_high_hz=args.search_high_hz,
            ppg_low_hz=args.ppg_low_hz,
            ppg_high_hz=args.ppg_high_hz,
            filter_order=args.filter_order,
        )
        ppg_fft = postprocess_by_record(
            ppg_fft,
            raw_col="ppg_fft_hr_raw_bpm",
            est_col="ppg_fft_hr_est_bpm",
            low_bpm=args.physio_low_bpm,
            high_bpm=args.physio_high_bpm,
            smooth_windows=args.smooth_windows,
            outlier_policy=args.outlier_policy,
        )
        desired, cleaned, artifact = make_nlms_cleaned_ppg(
            record,
            sig,
            fs=args.fs,
            ppg_low_hz=args.ppg_low_hz,
            ppg_high_hz=args.ppg_high_hz,
            filter_order=args.filter_order,
            nlms_filter_order=args.nlms_filter_order,
            nlms_mu=args.nlms_mu,
            nlms_eps=args.nlms_eps,
        )
        nlms = estimate_signal_fft_record(
            cleaned,
            bpm0,
            record,
            fs=args.fs,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            n_fft=args.n_fft,
            search_low_hz=args.search_low_hz,
            search_high_hz=args.search_high_hz,
            prefix="ppg_acc_nlms_fft",
        )
        nlms = postprocess_by_record(
            nlms,
            raw_col="ppg_acc_nlms_fft_hr_raw_bpm",
            est_col="ppg_acc_nlms_fft_hr_est_bpm",
            low_bpm=args.physio_low_bpm,
            high_bpm=args.physio_high_bpm,
            smooth_windows=args.smooth_windows,
            outlier_policy=args.outlier_policy,
        )
        frames.append(merge_on_window(ppg_fft, nlms))
        clean_summaries.append(
            {
                "record": record.name,
                "split": record.split,
                "desired_std": float(np.std(desired)),
                "cleaned_std": float(np.std(cleaned)),
                "artifact_std": float(np.std(artifact)),
                "artifact_to_desired_std_ratio": float(np.std(artifact) / max(np.std(desired), 1e-12)),
            }
        )

    windows = pd.concat(frames, ignore_index=True)
    methods = [
        ("PPG FFT", "ppg_fft_hr_est_bpm"),
        ("ACC-NLMS FFT", "ppg_acc_nlms_fft_hr_est_bpm"),
    ]
    windows = add_errors(windows, methods)
    metrics = build_metrics(windows, methods)

    windows_csv = args.outdir / "ppg_nlms_acc_fft_hr_windows.csv"
    metrics_csv = args.outdir / "ppg_nlms_acc_fft_metrics.csv"
    clean_csv = args.outdir / "nlms_cleaning_summary.csv"
    config_csv = args.outdir / "ppg_nlms_acc_fft_database_config.csv"
    windows.to_csv(windows_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
    pd.DataFrame(clean_summaries).to_csv(clean_csv, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "DATABASE",
                "data_dir": str(args.data_dir),
                "fs": args.fs,
                "window_sec": args.window_sec,
                "step_sec": args.step_sec,
                "n_fft": args.n_fft,
                "search_low_hz": args.search_low_hz,
                "search_high_hz": args.search_high_hz,
                "nlms_filter_order": args.nlms_filter_order,
                "nlms_mu": args.nlms_mu,
                "smooth_windows": args.smooth_windows,
            }
        ]
    ).to_csv(config_csv, index=False)

    plot_records = [] if args.plot_records == ["none"] else args.plot_records
    plot_paths = []
    for record_name in plot_records:
        if record_name in {record.name for record in records}:
            plot_paths.append(plot_time_series(windows, record_name, plots_dir))
    plot_paths.extend([plot_mae(metrics, plots_dir), plot_scatter(windows, metrics, plots_dir), plot_bland_altman(windows, metrics, plots_dir)])

    print(f"Wrote windows: {windows_csv}")
    print(f"Wrote metrics: {metrics_csv}")
    print(f"Wrote NLMS summary: {clean_csv}")
    print("Plots:")
    for path in plot_paths:
        print(f"  {path}")
    print(metrics[metrics["record"].str.startswith("ALL_")].to_string(index=False))


if __name__ == "__main__":
    main()
