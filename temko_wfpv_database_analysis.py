#!/usr/bin/env python3
"""Python reproduction of Temko WFPV on the TROIKA-style DATABASE folder.

This script follows the online MATLAB implementation in andtem2000/PPG
(`PPG_WFPV_TBME2017.m`) for records stored as MATLAB `.mat` files:

    two-channel PPG averaging -> PPG/ACC band-pass filtering -> downsample to 25 Hz
    -> Wiener spectral weighting -> phase-vocoder frequency refinement
    -> history-constrained HR tracking -> BPM0 error analysis.

Training records have shape 6 x N:
    ECG, PPG1, PPG2, ACC_X, ACC_Y, ACC_Z.

Competition/test records have shape 5 x N:
    PPG1, PPG2, ACC_X, ACC_Y, ACC_Z.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat


DEFAULT_DATA_DIR = Path("/Users/xiongzaizai/PPG/DATABASE")
DEFAULT_OUTDIR = Path("outputs_database_temko_wfpv")


@dataclass(frozen=True)
class DatabaseRecord:
    name: str
    split: str
    signal_path: Path
    truth_path: Path


def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=float)
    y = x.copy()
    if not np.all(finite):
        y[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    if std < np.finfo(float).eps:
        return np.zeros_like(y, dtype=float)
    return (y - float(np.mean(y))) / std


def safe_norm_spectrum(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = float(np.nanmax(np.abs(x))) if x.size else 0.0
    if not np.isfinite(scale) or scale < eps:
        return np.zeros_like(x, dtype=float)
    return x / scale


def moving_average_same(x: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if width <= 1 or x.size == 0:
        return x.copy()
    kernel = np.ones(width, dtype=float) / width
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def bandpass_lfilter(x: np.ndarray, fs: float, low_hz: float = 0.4, high_hz: float = 4.0) -> np.ndarray:
    b, a = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=fs)
    return signal.lfilter(b, a, np.asarray(x, dtype=float))


def phase_vocoder_frequencies(
    current_fft: np.ndarray,
    previous_fft: np.ndarray | None,
    bin_freqs: np.ndarray,
    hop_s: float,
    smooth_bins: int,
) -> np.ndarray:
    refined = bin_freqs.astype(float).copy()
    if previous_fft is not None and len(previous_fft) == len(current_fft):
        wraps = np.arange(20, dtype=float)
        for idx, bin_freq in enumerate(bin_freqs):
            phase_delta = np.angle(current_fft[idx]) - np.angle(previous_fft[idx])
            candidates = (phase_delta + 2.0 * np.pi * wraps) / (2.0 * np.pi * hop_s)
            refined[idx] = candidates[int(np.argmin(np.abs(candidates - bin_freq)))]
    return moving_average_same(refined, smooth_bins)


def strongest_within_range(spectrum: np.ndarray, allowed_idx: np.ndarray) -> int:
    if allowed_idx.size == 0:
        return int(np.nanargmax(spectrum))
    local_idx = int(np.nanargmax(spectrum[allowed_idx]))
    return int(allowed_idx[local_idx])


def discover_records(data_dir: Path, requested: list[str]) -> list[DatabaseRecord]:
    records: list[DatabaseRecord] = []

    train_dir = data_dir / "Training_data"
    for signal_path in sorted(train_dir.glob("DATA_*.mat")):
        if signal_path.name.endswith("_BPMtrace.mat"):
            continue
        name = signal_path.stem
        truth_path = signal_path.with_name(f"{name}_BPMtrace.mat")
        if truth_path.exists():
            records.append(DatabaseRecord(name=name, split="training", signal_path=signal_path, truth_path=truth_path))

    test_dir = data_dir / "Competition_data"
    for signal_path in sorted(test_dir.glob("TEST_*.mat")):
        name = signal_path.stem
        truth_path = signal_path.with_name(f"True{name[4:]}.mat")
        if truth_path.exists():
            records.append(DatabaseRecord(name=name, split="competition", signal_path=signal_path, truth_path=truth_path))

    if requested == ["all"]:
        return records

    by_name = {record.name: record for record in records}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown record(s): {', '.join(unknown)}")
    return [by_name[name] for name in requested]


def load_record(record: DatabaseRecord) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(record.signal_path)
    truth = loadmat(record.truth_path)
    if "sig" not in mat:
        raise KeyError(f"{record.signal_path} does not contain variable 'sig'")
    if "BPM0" not in truth:
        raise KeyError(f"{record.truth_path} does not contain variable 'BPM0'")
    sig = np.asarray(mat["sig"], dtype=float)
    bpm0 = np.asarray(truth["BPM0"], dtype=float).reshape(-1)
    return sig, bpm0


def extract_channels(record: DatabaseRecord, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sig.ndim != 2:
        raise ValueError(f"{record.name}: expected 2D sig array, got shape {sig.shape}")
    if record.split == "competition":
        if sig.shape[0] < 5:
            raise ValueError(f"{record.name}: competition record should have at least 5 channels")
        ch1, ch2, ch3, ch4, ch5 = 0, 1, 2, 3, 4
    else:
        if sig.shape[0] < 6:
            raise ValueError(f"{record.name}: training record should have at least 6 channels")
        ch1, ch2, ch3, ch4, ch5 = 1, 2, 3, 4, 5
    return sig[ch1], sig[ch2], sig[ch3], sig[ch4], sig[ch5]


def estimate_temko_wfpv(
    record: DatabaseRecord,
    sig: np.ndarray,
    bpm0: np.ndarray,
    fs: float,
    fft_res: int,
    wf_length: int,
    search_low_hz: float,
    search_high_hz: float,
    window_sec: float,
    step_sec: float,
    smooth_freq_bins: int,
) -> pd.DataFrame:
    ppg1, ppg2, acc_x, acc_y, acc_z = extract_channels(record, sig)
    window = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    window_nb = int(np.floor((sig.shape[1] - window) / step) + 1)
    window_nb = min(window_nb, len(bpm0))
    target_fs = fs / 5.0

    freq_range_all = np.linspace(0.0, target_fs, fft_res)
    low_idx = int(np.argmin(np.abs(freq_range_all - search_low_hz)))
    high_idx = int(np.argmin(np.abs(freq_range_all - search_high_hz)))
    if high_idx <= low_idx:
        raise ValueError("Invalid frequency range after FFT bin selection")
    freq_range = freq_range_all[low_idx : high_idx + 1]

    w1_history: list[np.ndarray] = []
    w2_history: list[np.ndarray] = []
    previous_ppg_fft: np.ndarray | None = None
    previous_estimates: list[float] = []
    previous_range: np.ndarray | None = None
    rows: list[dict[str, float | int | str]] = []

    for i in range(window_nb):
        cur = slice(i * step, i * step + window)
        cur_ppg1 = bandpass_lfilter(ppg1[cur], fs)
        cur_ppg2 = bandpass_lfilter(ppg2[cur], fs)
        cur_acc_x = bandpass_lfilter(acc_x[cur], fs)
        cur_acc_y = bandpass_lfilter(acc_y[cur], fs)
        cur_acc_z = bandpass_lfilter(acc_z[cur], fs)

        ppg_average = 0.5 * safe_zscore(cur_ppg1) + 0.5 * safe_zscore(cur_ppg2)
        ppg_average = ppg_average[::5]
        cur_acc_x = cur_acc_x[::5]
        cur_acc_y = cur_acc_y[::5]
        cur_acc_z = cur_acc_z[::5]

        ppg_fft = np.fft.fft(ppg_average, fft_res)[low_idx : high_idx + 1]
        acc_x_fft = np.fft.fft(cur_acc_x, fft_res)[low_idx : high_idx + 1]
        acc_y_fft = np.fft.fft(cur_acc_y, fft_res)[low_idx : high_idx + 1]
        acc_z_fft = np.fft.fft(cur_acc_z, fft_res)[low_idx : high_idx + 1]

        freq_range_ppg = phase_vocoder_frequencies(
            ppg_fft,
            previous_ppg_fft,
            freq_range,
            hop_s=step_sec,
            smooth_bins=smooth_freq_bins,
        )
        previous_ppg_fft = ppg_fft

        ppg_abs = np.abs(ppg_fft)
        ppg_norm = safe_norm_spectrum(ppg_abs)
        acc_x_norm = safe_norm_spectrum(np.abs(acc_x_fft))
        acc_y_norm = safe_norm_spectrum(np.abs(acc_y_fft))
        acc_z_norm = safe_norm_spectrum(np.abs(acc_z_fft))
        acc_mean_norm = (acc_x_norm + acc_y_norm + acc_z_norm) / 3.0

        w1_env = np.mean((w1_history + [ppg_norm])[-(wf_length + 1) :], axis=0)
        w1_env_norm = safe_norm_spectrum(w1_env)
        wf1 = 1.0 - acc_mean_norm / np.maximum(w1_env_norm, 1e-12)
        wf1[wf1 < 0] = -1.0
        w1_clean = ppg_abs * wf1

        # MATLAB updates W2_FFTi with the cleaned WF2 spectrum, so WF2 is
        # recursively estimated from past cleaned spectra rather than from
        # only raw PPG spectral envelopes.
        w2_env = np.mean((w2_history + [ppg_norm])[-(wf_length + 1) :], axis=0)
        w2_env_norm = safe_norm_spectrum(w2_env)
        wf2 = w2_env_norm / np.maximum(acc_mean_norm + w2_env_norm, 1e-12)
        w2_clean = ppg_abs * wf2

        w1_history.append(ppg_norm)
        w2_history.append(safe_norm_spectrum(w2_clean))

        w1_std = float(np.std(w1_clean, ddof=1)) if w1_clean.size > 1 else 0.0
        w2_std = float(np.std(w2_clean, ddof=1)) if w2_clean.size > 1 else 0.0
        if w1_std > np.finfo(float).eps:
            w1_clean = w1_clean / w1_std
        if w2_std > np.finfo(float).eps:
            w2_clean = w2_clean / w2_std

        clean_spectrum = w1_clean + w2_clean

        hist_int = 25.0
        warmup = 15 if record.split == "competition" or record.name == "DATA_S04_T01" else 30
        if len(previous_estimates) > warmup:
            diffs = np.abs(np.diff(previous_estimates))
            if diffs.size and np.isfinite(diffs).any():
                hist_int = float(np.nanmax(diffs) + 5.0)

        if previous_range is None or previous_range.size == 0:
            peak_idx = int(np.nanargmax(clean_spectrum))
        else:
            peak_idx = strongest_within_range(clean_spectrum, previous_range)

        peak_idx = max(0, min(peak_idx, len(freq_range) - 1))
        hr_est = float(freq_range_ppg[peak_idx] * 60.0)

        if len(previous_estimates) >= 5 and abs(hr_est - previous_estimates[-1]) > 5.0:
            prev = np.asarray(previous_estimates[-5:], dtype=float)
            x = np.arange(1, len(prev) + 1)
            slope, intercept = np.polyfit(x, prev, 1)
            predicted = float(slope * (len(prev) + 1) + intercept)
            hr_est = 0.8 * hr_est + 0.2 * predicted

        previous_estimates.append(hr_est)
        if len(previous_estimates) >= 2:
            recent = np.asarray(previous_estimates[max(1, len(previous_estimates) - 6) :], dtype=float)
            older = np.asarray(previous_estimates[max(0, len(previous_estimates) - 7) : -1], dtype=float)
            if len(recent) == len(older) and len(recent) > 0:
                hr_est = hr_est + float(np.sum(np.sign(recent - older)) * 0.1)
                previous_estimates[-1] = hr_est

        df_bpm = float(np.median(np.diff(freq_range)) * 60.0)
        half_width = int(round(hist_int / max(df_bpm, 1e-12)))
        previous_range = np.arange(max(0, peak_idx - half_width), min(len(freq_range), peak_idx + half_width + 1))

        start_s = i * step_sec
        rows.append(
            {
                "record": record.name,
                "split": record.split,
                "window_index": i + 1,
                "start_time_s": start_s,
                "end_time_s": start_s + window_sec,
                "center_time_s": start_s + window_sec / 2.0,
                "hr_true_bpm": float(bpm0[i]),
                "temko_wfpv_hr_bpm": hr_est,
                "temko_peak_hz": float(freq_range_ppg[peak_idx]),
                "temko_peak_power": float(clean_spectrum[peak_idx]),
                "tracking_low_bpm": float(freq_range[previous_range[0]] * 60.0) if previous_range.size else np.nan,
                "tracking_high_bpm": float(freq_range[previous_range[-1]] * 60.0) if previous_range.size else np.nan,
            }
        )

    return pd.DataFrame(rows)


def metric_row(df: pd.DataFrame, record_name: str, split: str) -> dict[str, float | int | str]:
    valid = df[["hr_true_bpm", "temko_wfpv_hr_bpm"]].dropna()
    row: dict[str, float | int | str] = {
        "record": record_name,
        "split": split,
        "n_windows": len(df),
        "n_valid": len(valid),
    }
    if valid.empty:
        row.update(
            {
                "mae_bpm": np.nan,
                "std_error_bpm": np.nan,
                "std_abs_error_bpm": np.nan,
                "mean_relative_error_percent": np.nan,
                "pearson_r": np.nan,
                "bias_bpm": np.nan,
                "bland_altman_lower_bpm": np.nan,
                "bland_altman_upper_bpm": np.nan,
            }
        )
        return row

    error = valid["temko_wfpv_hr_bpm"] - valid["hr_true_bpm"]
    abs_error = np.abs(error)
    std_error = float(error.std(ddof=1)) if len(error) >= 2 else np.nan
    bias = float(error.mean())
    row.update(
        {
            "mae_bpm": float(abs_error.mean()),
            "std_error_bpm": std_error,
            "std_abs_error_bpm": float(abs_error.std(ddof=1)) if len(abs_error) >= 2 else np.nan,
            "mean_relative_error_percent": float((abs_error / valid["hr_true_bpm"] * 100.0).mean()),
            "pearson_r": float(valid["hr_true_bpm"].corr(valid["temko_wfpv_hr_bpm"])) if len(valid) >= 2 else np.nan,
            "bias_bpm": bias,
            "bland_altman_lower_bpm": bias - 1.96 * std_error if np.isfinite(std_error) else np.nan,
            "bland_altman_upper_bpm": bias + 1.96 * std_error if np.isfinite(std_error) else np.nan,
        }
    )
    return row


def build_metrics(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record_name, record_df in windows.groupby("record", sort=False):
        split = str(record_df["split"].iloc[0])
        rows.append(metric_row(record_df, record_name, split))
    for split, split_df in windows.groupby("split", sort=False):
        rows.append(metric_row(split_df, f"ALL_{split}", split))
    rows.append(metric_row(windows, "ALL_DATABASE", "all"))
    return pd.DataFrame(rows)


def plot_time_series(windows: pd.DataFrame, record_name: str, outdir: Path) -> Path:
    df = windows[windows["record"] == record_name]
    fig, ax = plt.subplots(figsize=(11.5, 4.7), constrained_layout=True)
    ax.plot(df["center_time_s"] / 60.0, df["hr_true_bpm"], label="HR true", linewidth=1.6, color="#1f77b4")
    ax.plot(df["center_time_s"] / 60.0, df["temko_wfpv_hr_bpm"], label="Temko WFPV", linewidth=1.25, color="#8c564b")
    ax.set_title(f"Temko WFPV HR vs BPM0 ground truth: {record_name}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / f"{record_name}_timeseries_temko_wfpv.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mae(metrics: pd.DataFrame, outdir: Path) -> Path:
    per_record = metrics[~metrics["record"].str.startswith("ALL_")].copy()
    per_record = per_record.sort_values(["split", "record"])
    colors = per_record["split"].map({"training": "#1f77b4", "competition": "#ff7f0e"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
    ax.bar(per_record["record"], per_record["mae_bpm"], color=colors)
    ax.set_title("Temko WFPV MAE by DATABASE record")
    ax.set_ylabel("MAE (bpm)")
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    path = outdir / "mae_by_record_temko_wfpv.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter(windows: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 5.6), constrained_layout=True)
    for split, color in [("training", "#1f77b4"), ("competition", "#ff7f0e")]:
        df = windows[windows["split"] == split]
        ax.scatter(df["hr_true_bpm"], df["temko_wfpv_hr_bpm"], s=12, alpha=0.35, label=split, color=color)
    vals = np.concatenate([windows["hr_true_bpm"].to_numpy(), windows["temko_wfpv_hr_bpm"].to_numpy()])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo = float(vals.min() - 5)
        hi = float(vals.max() + 5)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    overall = metrics[metrics["record"] == "ALL_DATABASE"].iloc[0]
    ax.text(
        0.04,
        0.96,
        f"ALL: r={overall['pearson_r']:.3f}\\nMAE={overall['mae_bpm']:.2f} bpm",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.set_title("Temko WFPV correlation")
    ax.set_xlabel("HR true BPM0 (bpm)")
    ax.set_ylabel("HR estimated (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "scatter_temko_wfpv.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_bland_altman(windows: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    valid = windows[["hr_true_bpm", "temko_wfpv_hr_bpm"]].dropna()
    mean_hr = valid.mean(axis=1)
    diff = valid["temko_wfpv_hr_bpm"] - valid["hr_true_bpm"]
    overall = metrics[metrics["record"] == "ALL_DATABASE"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    ax.scatter(mean_hr, diff, s=12, alpha=0.35, color="#8c564b")
    bias = float(overall["bias_bpm"])
    lower = float(overall["bland_altman_lower_bpm"])
    upper = float(overall["bland_altman_upper_bpm"])
    ax.axhline(bias, color="#8c564b", linewidth=1.2, label="Bias")
    ax.axhline(lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
    ax.axhline(upper, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Bland-Altman: Temko WFPV")
    ax.set_xlabel("Mean HR (bpm)")
    ax.set_ylabel("HR est - HR true (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "bland_altman_temko_wfpv.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Temko WFPV on DATABASE .mat records.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--records", nargs="+", default=["all"])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fs", type=float, default=125.0)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--search-low-hz", type=float, default=1.0)
    parser.add_argument("--search-high-hz", type=float, default=3.0)
    parser.add_argument("--wf-length", type=int, default=15)
    parser.add_argument("--smooth-freq-bins", type=int, default=3)
    parser.add_argument("--plot-records", nargs="+", default=["DATA_01_TYPE01", "DATA_08_TYPE02", "TEST_S01_T01", "TEST_S07_T02"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.data_dir, args.records)
    frames = []
    for record in records:
        sig, bpm0 = load_record(record)
        frame = estimate_temko_wfpv(
            record,
            sig=sig,
            bpm0=bpm0,
            fs=args.fs,
            fft_res=args.n_fft,
            wf_length=args.wf_length,
            search_low_hz=args.search_low_hz,
            search_high_hz=args.search_high_hz,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            smooth_freq_bins=args.smooth_freq_bins,
        )
        frames.append(frame)

    windows = pd.concat(frames, ignore_index=True)
    windows["error_bpm"] = windows["temko_wfpv_hr_bpm"] - windows["hr_true_bpm"]
    windows["abs_error_bpm"] = np.abs(windows["error_bpm"])
    windows["relative_abs_error_percent"] = windows["abs_error_bpm"] / windows["hr_true_bpm"] * 100.0
    metrics = build_metrics(windows)

    windows_path = args.outdir / "temko_wfpv_database_windows.csv"
    metrics_path = args.outdir / "temko_wfpv_database_metrics.csv"
    config_path = args.outdir / "temko_wfpv_database_config.csv"
    windows.to_csv(windows_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "source_repository": "https://github.com/andtem2000/PPG",
                "source_matlab_file": "PPG_WFPV_TBME2017.m",
                "data_dir": str(args.data_dir),
                "records": ",".join(record.name for record in records),
                "fs_hz": args.fs,
                "window_sec": args.window_sec,
                "step_sec": args.step_sec,
                "target_fs_hz": args.fs / 5.0,
                "n_fft": args.n_fft,
                "search_low_hz": args.search_low_hz,
                "search_high_hz": args.search_high_hz,
                "wf_length": args.wf_length,
            }
        ]
    ).to_csv(config_path, index=False)

    plot_paths = [
        plot_mae(metrics, plots_dir),
        plot_scatter(windows, metrics, plots_dir),
        plot_bland_altman(windows, metrics, plots_dir),
    ]
    available = set(windows["record"])
    for record_name in args.plot_records:
        if record_name in available:
            plot_paths.append(plot_time_series(windows, record_name, plots_dir))

    print(f"Wrote windows: {windows_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote config: {config_path}")
    print("Plots:")
    for path in plot_paths:
        print(f"  {path}")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
