#!/usr/bin/env python3
"""DATABASE PPG-only FFT baseline against BPM0 ground truth.

This script supersedes the earlier wrist-ppg-during-exercise version. It reads
the local TROIKA-style DATABASE folder:

Training records:
    sig = [ECG, PPG1, PPG2, ACC_X, ACC_Y, ACC_Z]

Competition records:
    sig = [PPG1, PPG2, ACC_X, ACC_Y, ACC_Z]

BPM0 is used as the ECG-derived window-level ground truth.
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
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat


DEFAULT_DATA_DIR = Path("/Users/xiongzaizai/PPG/DATABASE")
DEFAULT_OUTDIR = Path("outputs_database_ppg_fft_hr")


@dataclass(frozen=True)
class DatabaseRecord:
    name: str
    split: str
    signal_path: Path
    truth_path: Path


def discover_records(data_dir: Path, requested: list[str]) -> list[DatabaseRecord]:
    records: list[DatabaseRecord] = []

    train_dir = data_dir / "Training_data"
    for signal_path in sorted(train_dir.glob("DATA_*.mat")):
        if signal_path.name.endswith("_BPMtrace.mat"):
            continue
        truth_path = signal_path.with_name(f"{signal_path.stem}_BPMtrace.mat")
        if truth_path.exists():
            records.append(DatabaseRecord(signal_path.stem, "training", signal_path, truth_path))

    test_dir = data_dir / "Competition_data"
    for signal_path in sorted(test_dir.glob("TEST_*.mat")):
        truth_path = signal_path.with_name(f"True{signal_path.stem[4:]}.mat")
        if truth_path.exists():
            records.append(DatabaseRecord(signal_path.stem, "competition", signal_path, truth_path))

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


def extract_ppg_channels(record: DatabaseRecord, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sig.ndim != 2:
        raise ValueError(f"{record.name}: expected 2D sig array, got {sig.shape}")
    if record.split == "competition":
        if sig.shape[0] < 2:
            raise ValueError(f"{record.name}: competition record should contain PPG1/PPG2")
        return sig[0], sig[1]
    if sig.shape[0] < 3:
        raise ValueError(f"{record.name}: training record should contain ECG/PPG1/PPG2")
    return sig[1], sig[2]


def extract_acc_channels(record: DatabaseRecord, sig: np.ndarray) -> np.ndarray:
    if record.split == "competition":
        if sig.shape[0] < 5:
            raise ValueError(f"{record.name}: competition record should contain PPG1/PPG2/ACC_XYZ")
        return sig[2:5]
    if sig.shape[0] < 6:
        raise ValueError(f"{record.name}: training record should contain ECG/PPG1/PPG2/ACC_XYZ")
    return sig[3:6]

"""
These codes are used to preprocessing
Each PPG Window is first checked for non-finite values and bandpass filtered between 0.4 and 4 Hz (24 to 240 bpm) 
using a 4th-order Butterworth filter. 
The two PPG channels are then standardised separately within each window 
so that amplitude differences do not cause one channel to dominate the subsequent fusion.
"""

def fill_nan_1d(x: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(x)
    if np.all(finite):
        return x
    if not np.any(finite):
        return np.full_like(x, fill_value, dtype=float)
    x[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), x[finite])
    return x


def zscore(x: np.ndarray) -> np.ndarray:
    x = fill_nan_1d(x)
    std = float(np.std(x))
    if std < np.finfo(float).eps:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / std


def bandpass_signal(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    x = fill_nan_1d(x)
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    if x.size > 3 * max(len(sos), 1):
        return signal.sosfiltfilt(sos, x)
    return signal.sosfilt(sos, x)


def preprocess_ppg_window(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    return zscore(bandpass_signal(x, fs=fs, low_hz=low_hz, high_hz=high_hz, order=order))

"""Fast Fourier Transform (FFT) Based Herat Rate Estimation
The fused PPG window is demeaned and multiplied by a Hann window to reduce spectral laeakage.
A one-sided FFT with zero-padding to 4096 points is then computed. 
The frequency with the largest spectral power between 0.8 and3.0 Hz is selected and converted to beats per minute.

This baseline relies only on dominant spectral peak selection and 
therefore cannot distinguish a cardiac peak from a stronger motion-artifact peak.
"""

def fft_peak_hr(
    x: np.ndarray,
    fs: float,
    n_fft: int,
    low_hz: float,
    high_hz: float,
) -> tuple[float, float, float]:
    x = fill_nan_1d(x)
    x = x - float(np.mean(x))
    freqs = rfftfreq(n_fft, d=1.0 / fs)
    power = np.abs(rfft(x * np.hanning(len(x)), n=n_fft)) ** 2
    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return np.nan, np.nan, np.nan
    band_power = power[band]
    if band_power.size == 0 or not np.isfinite(band_power).any():
        return np.nan, np.nan, np.nan
    idx = int(np.nanargmax(band_power))
    peak_hz = float(freqs[band][idx])
    return peak_hz, peak_hz * 60.0, float(band_power[idx])

"""Window-Based Processing
Each recording is divided into 8s windows with a 2s step, corresponding to a 75% overlap at 125 Hz.
Each window is processed independently and aligned with the corresponding BPM0 reference value."""

def estimate_ppg_fft_record(
    record: DatabaseRecord,
    sig: np.ndarray,
    bpm0: np.ndarray,
    fs: float,
    window_sec: float,
    step_sec: float,
    n_fft: int,
    search_low_hz: float,
    search_high_hz: float,
    ppg_low_hz: float,
    ppg_high_hz: float,
    filter_order: int,
) -> pd.DataFrame:
    ppg1, ppg2 = extract_ppg_channels(record, sig)
    window = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    window_nb = int(np.floor((sig.shape[1] - window) / step) + 1)
    window_nb = min(window_nb, len(bpm0))
    rows = []

    for idx in range(window_nb):
        cur = slice(idx * step, idx * step + window)
        ppg1_proc = preprocess_ppg_window(ppg1[cur], fs, ppg_low_hz, ppg_high_hz, filter_order)
        ppg2_proc = preprocess_ppg_window(ppg2[cur], fs, ppg_low_hz, ppg_high_hz, filter_order)
        """The two preprocessed PPG channels are fused by equal-weight averaging.
        Since both channels are standardised beforehand, 
        the averaging is not dominated solely by differences in their original amplitudes scales."""
        ppg_mean = 0.5 * ppg1_proc + 0.5 * ppg2_proc
        peak_hz, hr_raw, peak_power = fft_peak_hr(
            ppg_mean,
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
                "ppg_fft_peak_hz_raw": peak_hz,
                "ppg_fft_hr_raw_bpm": hr_raw,
                "ppg_fft_peak_power_raw": peak_power,
            }
        )
    return pd.DataFrame(rows)

"""Postprocessing
Post-processing is performed separately for each recording.
Estimates outside the predefined physiological range are marked as invalid and, 
under the default setting, internal missing values are linearly interpolated. 
A trailing moving average over the current and four preceding windows is then applied to reduce short-term fluctuations.
"""

def postprocess_by_record(
    df: pd.DataFrame,
    raw_col: str,
    est_col: str,
    low_bpm: float,
    high_bpm: float,
    smooth_windows: int,
    outlier_policy: str,
) -> pd.DataFrame:
    pieces = []
    for _, record_df in df.groupby("record", sort=False):
        record_df = record_df.copy()
        raw = record_df[raw_col]
        outlier = raw.notna() & ((raw < low_bpm) | (raw > high_bpm))
        clean = raw.mask(outlier)
        if outlier_policy == "interpolate":
            clean = clean.interpolate(method="linear", limit_area="inside")
        elif outlier_policy == "previous":
            clean = clean.ffill()
        elif outlier_policy == "nan":
            pass
        else:
            raise ValueError(f"Unknown outlier policy: {outlier_policy}")
        if smooth_windows > 1:
            est = clean.rolling(smooth_windows, min_periods=1).mean()
        else:
            est = clean
        record_df[f"{raw_col}_outlier"] = outlier.to_numpy()
        record_df[f"{raw_col}_clean"] = clean.to_numpy()
        record_df[est_col] = est.to_numpy()
        pieces.append(record_df)
    return pd.concat(pieces, ignore_index=True)

"""Performance Evaluation
The final HR estimates are compared with the provided BPM0 reference values.
The script calculates avAE, avRE, sdAE, Pearson correlation coefficient, and Bland-Altman bias, and the corresponding 95%
limits of agreement."""

def method_metrics(df: pd.DataFrame, record_name: str, split: str, method: str, est_col: str) -> dict:
    valid = df[["hr_true_bpm", est_col]].dropna()
    row: dict[str, float | int | str] = {
        "record": record_name,
        "split": split,
        "method": method,
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
    error = valid[est_col] - valid["hr_true_bpm"]
    abs_error = np.abs(error)
    std_error = float(error.std(ddof=1)) if len(error) >= 2 else np.nan
    bias = float(error.mean())
    row.update(
        {
            "mae_bpm": float(abs_error.mean()),
            "std_error_bpm": std_error,
            "std_abs_error_bpm": float(abs_error.std(ddof=1)) if len(abs_error) >= 2 else np.nan,
            "mean_relative_error_percent": float((abs_error / valid["hr_true_bpm"] * 100.0).mean()),
            "pearson_r": float(valid["hr_true_bpm"].corr(valid[est_col])) if len(valid) >= 2 else np.nan,
            "bias_bpm": bias,
            "bland_altman_lower_bpm": bias - 1.96 * std_error if np.isfinite(std_error) else np.nan,
            "bland_altman_upper_bpm": bias + 1.96 * std_error if np.isfinite(std_error) else np.nan,
        }
    )
    return row


def build_metrics(df: pd.DataFrame, methods: list[tuple[str, str]]) -> pd.DataFrame:
    per_record_rows = []
    for record_name, record_df in df.groupby("record", sort=False):
        split = str(record_df["split"].iloc[0])
        for method, est_col in methods:
            if est_col in record_df:
                row = method_metrics(record_df, record_name, split, method, est_col)
                row["aggregation"] = "per_record_windows"
                row["temko_report_label"] = ""
                row["n_records"] = 1
                row["pooled_window_mae_bpm"] = row["mae_bpm"]
                row["pooled_window_std_abs_error_bpm"] = row["std_abs_error_bpm"]
                row["pooled_window_mean_relative_error_percent"] = row["mean_relative_error_percent"]
                per_record_rows.append(row)

    per_record = pd.DataFrame(per_record_rows)
    rows = per_record_rows.copy()

    for split, split_df in df.groupby("split", sort=False):
        for method, est_col in methods:
            if est_col in split_df:
                scope = per_record[
                    (per_record["split"] == split) & (per_record["method"] == method)
                ]
                report_label = (
                    "Err12"
                    if split == "training" and len(scope) == 12
                    else f"Err{len(scope)}_{split}_available"
                )
                rows.append(
                    record_equal_summary(
                        split_df,
                        scope,
                        f"ALL_{split}",
                        split,
                        method,
                        est_col,
                        report_label,
                    )
                )

    for method, est_col in methods:
        if est_col in df:
            scope = per_record[per_record["method"] == method]
            rows.append(
                record_equal_summary(
                    df,
                    scope,
                    "ALL_DATABASE",
                    "all",
                    method,
                    est_col,
                    f"ErrAll{len(scope)}_available",
                )
            )
    return pd.DataFrame(rows)


def record_equal_summary(
    windows: pd.DataFrame,
    per_record: pd.DataFrame,
    record_name: str,
    split: str,
    method: str,
    est_col: str,
    report_label: str,
) -> dict:
    """Use Temko-style record-equal errors while retaining pooled agreement metrics."""
    row = method_metrics(windows, record_name, split, method, est_col)
    row["aggregation"] = "record_equal_errors; pooled_association_and_agreement"
    row["temko_report_label"] = report_label
    row["n_records"] = int(len(per_record))
    row["pooled_window_mae_bpm"] = row["mae_bpm"]
    row["pooled_window_std_abs_error_bpm"] = row["std_abs_error_bpm"]
    row["pooled_window_mean_relative_error_percent"] = row["mean_relative_error_percent"]

    if not per_record.empty:
        row["mae_bpm"] = float(per_record["mae_bpm"].mean())
        row["std_abs_error_bpm"] = float(per_record["std_abs_error_bpm"].mean())
        row["mean_relative_error_percent"] = float(
            per_record["mean_relative_error_percent"].mean()
        )
    return row


def plot_time_series(df: pd.DataFrame, record_name: str, outdir: Path) -> Path:
    record_df = df[df["record"] == record_name]
    metric = method_metrics(record_df, record_name, str(record_df["split"].iloc[0]), "PPG FFT", "ppg_fft_hr_est_bpm")
    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    ax.plot(
        record_df["center_time_s"] / 60.0,
        record_df["hr_true_bpm"],
        label="BPM0 true HR",
        linewidth=2.0,
        color="#222222",
    )
    ax.plot(
        record_df["center_time_s"] / 60.0,
        record_df["ppg_fft_hr_est_bpm"],
        label="PPG FFT",
        linewidth=1.45,
        color="#0072B2",
    )
    ax.set_title(f"DATABASE PPG FFT HR vs BPM0: {record_name}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.985,
        0.06,
        f"MAE={metric['mae_bpm']:.2f} bpm\n"
        f"sdAE={metric['std_abs_error_bpm']:.2f} bpm\n"
        f"r={metric['pearson_r']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88, "pad": 4},
    )
    path = outdir / f"{record_name}_timeseries_ppg_fft_vs_bpm0.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    valid = df[["hr_true_bpm", "ppg_fft_hr_est_bpm"]].dropna()
    row = metrics[(metrics["record"] == "ALL_DATABASE") & (metrics["method"] == "PPG FFT")].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    ax.scatter(valid["hr_true_bpm"], valid["ppg_fft_hr_est_bpm"], s=12, alpha=0.35, color="#0072B2")
    lo = float(valid.min().min() - 5)
    hi = float(valid.max().max() + 5)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.text(
        0.04,
        0.96,
        f"r={row['pearson_r']:.3f}\navAE_rec={row['mae_bpm']:.2f} bpm",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_title("DATABASE PPG FFT HR correlation")
    ax.set_xlabel("BPM0 true HR (bpm)")
    ax.set_ylabel("PPG FFT HR (bpm)")
    ax.grid(True, alpha=0.25)
    path = outdir / "scatter_ppg_fft_vs_bpm0.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_bland_altman(df: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    valid = df[["hr_true_bpm", "ppg_fft_hr_est_bpm"]].dropna()
    mean_hr = valid.mean(axis=1)
    diff_hr = valid["ppg_fft_hr_est_bpm"] - valid["hr_true_bpm"]
    row = metrics[(metrics["record"] == "ALL_DATABASE") & (metrics["method"] == "PPG FFT")].iloc[0]
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    ax.scatter(mean_hr, diff_hr, s=12, alpha=0.35)
    bias = float(row["bias_bpm"])
    lower = float(row["bland_altman_lower_bpm"])
    upper = float(row["bland_altman_upper_bpm"])
    ax.axhline(bias, color="#d62728", linewidth=1.2, label="Bias")
    ax.axhline(lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
    ax.axhline(upper, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Bland-Altman: DATABASE PPG FFT")
    ax.set_xlabel("Mean HR (bpm)")
    ax.set_ylabel("HR est - BPM0 (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "bland_altman_ppg_fft_vs_bpm0.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPG-only FFT HR estimation on DATABASE.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--records", nargs="+", default=["all"])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fs", type=float, default=125.0)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--step-sec", type=float, default=2.0)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--search-low-hz", type=float, default=1.0)
    parser.add_argument("--search-high-hz", type=float, default=3.0)
    parser.add_argument("--ppg-low-hz", type=float, default=0.4)
    parser.add_argument("--ppg-high-hz", type=float, default=4.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--physio-low-bpm", type=float, default=40.0)
    parser.add_argument("--physio-high-bpm", type=float, default=220.0)
    parser.add_argument("--smooth-windows", type=int, default=5)
    parser.add_argument("--outlier-policy", choices=["interpolate", "previous", "nan"], default="interpolate")
    parser.add_argument(
        "--plot-records",
        nargs="+",
        default=["DATA_01_TYPE01", "DATA_08_TYPE02", "DATA_10_TYPE02", "TEST_S01_T01", "TEST_S07_T02"],
    )
    return parser.parse_args()

""" Complete PPG-only FFT HR estimation pipeline for DATABASE. 
The main function loads all selected recordings, applies the complete PPG-only FFT pipline, performs record-wise 
post-processing, calculates the evaluation metrics, and exports the numerical results and figures."""
def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.data_dir, args.records)
    frames = []
    for record in records:
        sig, bpm0 = load_record(record)
        frames.append(
            estimate_ppg_fft_record(
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
        )

    windows = pd.concat(frames, ignore_index=True)
    windows = postprocess_by_record(
        windows,
        raw_col="ppg_fft_hr_raw_bpm",
        est_col="ppg_fft_hr_est_bpm",
        low_bpm=args.physio_low_bpm,
        high_bpm=args.physio_high_bpm,
        smooth_windows=args.smooth_windows,
        outlier_policy=args.outlier_policy,
    )
    windows["error_bpm"] = windows["ppg_fft_hr_est_bpm"] - windows["hr_true_bpm"]
    windows["abs_error_bpm"] = np.abs(windows["error_bpm"])
    windows["relative_abs_error_percent"] = windows["abs_error_bpm"] / windows["hr_true_bpm"] * 100.0
    metrics = build_metrics(windows, [("PPG FFT", "ppg_fft_hr_est_bpm")])

    windows_csv = args.outdir / "ppg_fft_hr_windows.csv"
    metrics_csv = args.outdir / "ppg_fft_metrics.csv"
    config_csv = args.outdir / "ppg_fft_database_config.csv"
    windows.to_csv(windows_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
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
                "smooth_windows": args.smooth_windows,
                "error_summary_aggregation": "record_equal_as_in_Temko_MATLAB",
            }
        ]
    ).to_csv(config_csv, index=False)

    plot_records = [] if args.plot_records == ["none"] else args.plot_records
    plot_paths = []
    for record_name in plot_records:
        if record_name in {record.name for record in records}:
            plot_paths.append(plot_time_series(windows, record_name, plots_dir))
    plot_paths.append(plot_scatter(windows, metrics, plots_dir))
    plot_paths.append(plot_bland_altman(windows, metrics, plots_dir))

    print(f"Wrote windows: {windows_csv}")
    print(f"Wrote metrics: {metrics_csv}")
    print("Plots:")
    for path in plot_paths:
        print(f"  {path}")
    print(metrics[metrics["record"].str.startswith("ALL_")].to_string(index=False))


if __name__ == "__main__":
    main()
