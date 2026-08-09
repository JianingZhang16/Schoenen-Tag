#!/usr/bin/env python3
"""Offline Temko WFPV + Viterbi decoding on the TROIKA-style DATABASE folder.

This script follows the offline MATLAB implementation in andtem2000/PPG
(`PPG_WFPV_VD_TBME2017_offline.m`). It first computes the WFPV spectral
emission matrix for each 8 s / 2 s window, estimates a leave-one-record-out
transition matrix from BPM0 ground-truth traces, and then uses Viterbi decoding
to obtain a globally consistent HR track.

The output HR used for metrics is the 4-window moving-smoothed Viterbi track,
matching the MATLAB error calculation:

    mean(abs(BPM0 - moving(BPM_est', 4)))
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

from temko_wfpv_database_analysis import (
    DEFAULT_DATA_DIR,
    DatabaseRecord,
    bandpass_lfilter,
    discover_records,
    extract_channels,
    load_record,
    phase_vocoder_frequencies,
    safe_norm_spectrum,
)


DEFAULT_OUTDIR = Path("outputs_database_temko_wfpv_vd")
"""
Store the complete WFPV spectral observations and phase-vocoder-refined
frequency grids required by the later offline Viterbi decoding stage.
"""
@dataclass(frozen=True)
class WfpvEmission:
    record: DatabaseRecord
    bpm0: np.ndarray
    center_time_s: np.ndarray
    freq_range_hz: np.ndarray
    freq_range_ppg_hz: np.ndarray
    emission: np.ndarray


def matlab_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def matlab_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    y = x.copy()
    if not np.all(finite):
        y[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    std = matlab_std(y)
    if std < np.finfo(float).eps:
        return np.zeros_like(y)
    return (y - float(np.mean(y))) / std

"""Reproduce the MATLAB moving.m operation used in Temko's offline code.
It is later applied to the final Viterbi-decoded HR trajectory."""
def matlab_moving_1d(x: np.ndarray, width: int) -> np.ndarray:
    """Reproduce Aslak Grinsted's MATLAB moving.m for a 1D vector."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if width <= 1 or x.size == 0:
        return x.copy()
    if width > x.size:
        raise ValueError("moving-average width must not exceed the input length")
    filt = np.ones(width, dtype=float) / width
    y = signal.lfilter(filt, [1.0], x)
    n = x.size
    is_odd = width & 1
    m2 = width // 2
    idx = (
        [width - 1] * (m2 - 1 + is_odd)
        + list(range(width - 1, n))
        + [n - 1] * m2
    )
    return y[np.asarray(idx, dtype=int)]

"""Apply the MATLAB moving operation independently to each matrix column.
his is used to smooth the estimated transition probabilities."""
def matlab_moving_axis0(x: np.ndarray, width: int) -> np.ndarray:
    """Apply MATLAB moving.m independently along axis 0."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return matlab_moving_1d(x, width)
    cols = [matlab_moving_1d(x[:, col], width) for col in range(x.shape[1])]
    return np.column_stack(cols)

"""Apply the same preprocessing, phase-vocoder refinement, and Wiener spectral
attenuation as in the online WFPV method. Instead of selecting one HR peak
immediately, retain the complete cleaned spectrum of every window as the
emission evidence for offline decoding.
"""
def compute_wfpv_emission(
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
) -> WfpvEmission:
    ppg1, ppg2, acc_x, acc_y, acc_z = extract_channels(record, sig)
    window = int(round(window_sec * fs))
    step = int(round(step_sec * fs))
    window_nb = int(np.floor((sig.shape[1] - window) / step) + 1)
    if window_nb != len(bpm0):
        raise ValueError(
            f"{record.name}: signal yields {window_nb} windows but BPM0 contains {len(bpm0)} values"
        )
    target_fs = fs / 5.0

    freq_range_all = np.linspace(0.0, target_fs, fft_res)
    low_idx = int(np.argmin(np.abs(freq_range_all - search_low_hz)))
    high_idx = int(np.argmin(np.abs(freq_range_all - search_high_hz)))
    if high_idx <= low_idx:
        raise ValueError("Invalid frequency range after FFT bin selection")
    freq_range = freq_range_all[low_idx : high_idx + 1]

    previous_ppg_fft: np.ndarray | None = None
    w1_history: list[np.ndarray] = []
    w2_history: list[np.ndarray] = []
    emissions: list[np.ndarray] = []
    freq_range_ppg_rows: list[np.ndarray] = []
    centers: list[float] = []

    for i in range(window_nb):
        cur = slice(i * step, i * step + window)
        cur_ppg1 = bandpass_lfilter(ppg1[cur], fs)
        cur_ppg2 = bandpass_lfilter(ppg2[cur], fs)
        cur_acc_x = bandpass_lfilter(acc_x[cur], fs)
        cur_acc_y = bandpass_lfilter(acc_y[cur], fs)
        cur_acc_z = bandpass_lfilter(acc_z[cur], fs)

        ppg_average = 0.5 * matlab_zscore(cur_ppg1) + 0.5 * matlab_zscore(cur_ppg2)
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
            smooth_bins=1,
        )
        freq_range_ppg = matlab_moving_1d(freq_range_ppg, smooth_freq_bins)
        freq_range_ppg_rows.append(freq_range_ppg)
        previous_ppg_fft = ppg_fft

        ppg_abs = np.abs(ppg_fft)
        ppg_norm = safe_norm_spectrum(ppg_abs)

        w1_env = np.mean((w1_history + [ppg_norm])[-(wf_length + 1) :], axis=0)
        w1_env_norm = safe_norm_spectrum(w1_env)
        acc_x_norm = safe_norm_spectrum(np.abs(acc_x_fft))
        acc_y_norm = safe_norm_spectrum(np.abs(acc_y_fft))
        acc_z_norm = safe_norm_spectrum(np.abs(acc_z_fft))
        acc_mean_norm = (acc_x_norm + acc_y_norm + acc_z_norm) / 3.0
        wf1 = 1.0 - acc_mean_norm / np.maximum(w1_env_norm, 1e-12)
        wf1[wf1 < 0] = -1.0
        w1_clean = ppg_abs * wf1

        w2_env = np.mean((w2_history + [ppg_norm])[-(wf_length + 1) :], axis=0)
        w2_env_norm = safe_norm_spectrum(w2_env)
        wf2 = w2_env_norm / np.maximum(acc_mean_norm + w2_env_norm, 1e-12)
        w2_clean = ppg_abs * wf2
        w2_history.append(safe_norm_spectrum(w2_clean))
        w1_history.append(ppg_norm)

        w1_std = matlab_std(w1_clean)
        w2_std = matlab_std(w2_clean)
        if w1_std > np.finfo(float).eps:
            w1_clean = w1_clean / w1_std
        if w2_std > np.finfo(float).eps:
            w2_clean = w2_clean / w2_std

            """Store the combined WF1 and WF2 spectrum as the emission evidence for all
            possible HR states in the current window."""

        emissions.append(w1_clean + w2_clean)
        centers.append(i * step_sec + window_sec / 2.0)

    return WfpvEmission(
        record=record,
        bpm0=np.asarray(bpm0[:window_nb], dtype=float),
        center_time_s=np.asarray(centers, dtype=float),
        freq_range_hz=freq_range,
        freq_range_ppg_hz=np.vstack(freq_range_ppg_rows),
        emission=np.vstack(emissions),
    )

"""Estimate the probabilities of transitions between consecutive HR states from
the BPM0 trajectories of all records except the record currently being
decoded. This implements the leave-one-record-out procedure.
"""
def transition_matrix_from_truth(
    emissions_by_record: dict[str, WfpvEmission],
    held_out_name: str,
    freq_range_hz: np.ndarray,
) -> np.ndarray:
    num_states = len(freq_range_hz)
    """Divide the HR interval from 60 to 180 BPM into the same number of discrete
    states as the frequency bins in the WFPV emission spectrum."""
    edges = np.linspace(60.0, 180.0, num_states + 1)
    trans = np.zeros((num_states, num_states), dtype=float)

    """Use only the reference traces of the remaining records when estimating the
    transition probabilities for the current held-out record.
    """

    for name, emission in emissions_by_record.items():
        if name == held_out_name:
            continue
        bpm0 = np.asarray(emission.bpm0, dtype=float)
        if bpm0.size < 2:
            continue
        """Map each BPM0 value to its corresponding discrete HR state."""
        bins = np.searchsorted(edges, bpm0, side="right")
        bins[bins == 0] = 1
        bins[bins > num_states] = num_states
        bins = bins.astype(int) - 1
        """Count the transitions between consecutive reference HR states."""
        for src, dst in zip(bins[:-1], bins[1:]):
            trans[src, dst] += 1.0

            """Normalise each row to obtain the conditional probability of the next state
            given the current HR state."""

    row_sums = trans.sum(axis=1, keepdims=True)
    out = np.divide(trans, row_sums, out=np.full_like(trans, np.finfo(float).eps), where=row_sums > 0)
    out[~np.isfinite(out)] = np.finfo(float).eps
    return out

"""Determine the globally best HR-state path by combining the WFPV spectral
evidence with the learned transition probabilities over the complete record."""
def viterbi_path(transition: np.ndarray, emission: np.ndarray) -> np.ndarray:
    """Replicate viterbi_path2.m using log transition and raw emission values."""
    e = np.asarray(emission, dtype=float)
    if e.ndim != 2:
        raise ValueError("emission must be states x frames")
    num_states, num_frames = e.shape
    if num_frames == 0:
        return np.asarray([], dtype=int)

    """Convert the transition probabilities to the logarithmic domain for the
    dynamic-programming recursion."""

    with np.errstate(divide="ignore"):
        log_tr = np.log(np.asarray(transition, dtype=float))
    log_tr[~np.isfinite(log_tr)] = -np.inf

    """Store the best predecessor of every state at each window for later backtracking to obtain the optimal HR-state path.
    """

    ptr = np.zeros((num_states, num_frames), dtype=int)
    diag = np.diag(log_tr)
    v_old = diag * e[:, 0]
    v = np.full(num_states, -np.inf, dtype=float)

    """Recursively calculate the best cumulative score for every HR state at each
    analysis window."""

    for count in range(num_frames):
        for state in range(num_states):
            vals = v_old + log_tr[:, state]
            best_prev = int(np.argmax(vals))
            ptr[state, count] = best_prev
            v[state] = e[state, count] + vals[best_prev]
        v_old = v.copy()

        """Backtrack from the highest-scoring final state to recover the complete
        globally consistent HR-state path.
        """

    path = np.zeros(num_frames, dtype=int)
    path[-1] = int(np.argmax(v))
    for count in range(num_frames - 2, -1, -1):
        path[count] = ptr[path[count + 1], count + 1]
    return path

"""Decode each recording using a transition model estimated from the remaining
records, convert the decoded states to BPM, and apply the final temporal 
smoothing used by Temko's offline method.
"""
def estimate_offline_vd(emissions_by_record: dict[str, WfpvEmission]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for name, emission in emissions_by_record.items():
        transition = transition_matrix_from_truth(emissions_by_record, name, emission.freq_range_hz)
        """Smooth the transition probabilities using the four-point MATLAB moving
        operation before Viterbi decoding.
        """
        transition = matlab_moving_axis0(transition.T, 4).T
        """Apply Viterbi decoding. The emission matrix is transposed so that rows
        represent HR states and columns represent analysis windows.
        """
        state_path = viterbi_path(transition, emission.emission.T)
        """Convert each decoded state to BPM using the phase-vocoder-refined frequency
        grid of the corresponding analysis window.
        """
        raw_hr = np.asarray(
            [emission.freq_range_ppg_hz[i, state] * 60.0 for i, state in enumerate(state_path)],
            dtype=float,
        )
        """Apply the four-point MATLAB moving operation to the decoded HR trajectory.
        The smoothed sequence is used as the final offline WFPV+VD estimate.
        """
        smoothed_hr = matlab_moving_1d(raw_hr, 4)

        """Store the reference HR, raw Viterbi estimate, smoothed final estimate, and
        decoded state for every analysis window."""

        for i, (truth, raw, smooth, state) in enumerate(zip(emission.bpm0, raw_hr, smoothed_hr, state_path)):
            center_s = float(emission.center_time_s[i])
            rows.append(
                {
                    "record": emission.record.name,
                    "split": emission.record.split,
                    "window_index": i + 1,
                    "center_time_s": center_s,
                    "start_time_s": center_s - 4.0,
                    "end_time_s": center_s + 4.0,
                    "hr_true_bpm": float(truth),
                    "temko_wfpv_vd_hr_bpm": float(smooth),
                    "temko_wfpv_vd_raw_hr_bpm": float(raw),
                    "viterbi_state": int(state + 1),
                    "viterbi_state_freq_hz": float(emission.freq_range_hz[state]),
                }
            )

    windows = pd.DataFrame(rows)
    windows["error_bpm"] = windows["temko_wfpv_vd_hr_bpm"] - windows["hr_true_bpm"]
    windows["abs_error_bpm"] = np.abs(windows["error_bpm"])
    windows["relative_abs_error_percent"] = windows["abs_error_bpm"] / windows["hr_true_bpm"] * 100.0
    return windows

"""Compare the final offline WFPV+VD estimates with BPM0 and calculate avAE,
avRE, sdAE, Pearson correlation, Bland--Altman bias, and limits of agreement.
"""
def metric_row(df: pd.DataFrame, record_name: str, split: str) -> dict[str, float | int | str]:
    valid = df[["hr_true_bpm", "temko_wfpv_vd_hr_bpm"]].dropna()
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

    error = valid["temko_wfpv_vd_hr_bpm"] - valid["hr_true_bpm"]
    abs_error = np.abs(error)
    std_error = float(error.std(ddof=1)) if len(error) >= 2 else np.nan
    bias = float(error.mean())
    row.update(
        {
            "mae_bpm": float(abs_error.mean()),
            "std_error_bpm": std_error,
            "std_abs_error_bpm": float(abs_error.std(ddof=1)) if len(abs_error) >= 2 else np.nan,
            "mean_relative_error_percent": float((abs_error / valid["hr_true_bpm"] * 100.0).mean()),
            "pearson_r": float(valid["hr_true_bpm"].corr(valid["temko_wfpv_vd_hr_bpm"])) if len(valid) >= 2 else np.nan,
            "bias_bpm": bias,
            "bland_altman_lower_bpm": bias - 1.96 * std_error if np.isfinite(std_error) else np.nan,
            "bland_altman_upper_bpm": bias + 1.96 * std_error if np.isfinite(std_error) else np.nan,
        }
    )
    return row


def record_equal_summary(
    windows: pd.DataFrame,
    per_record: pd.DataFrame,
    record_name: str,
    split: str,
    temko_report_label: str,
) -> dict[str, float | int | str]:
    """Aggregate error metrics with equal weight per recording, as in Temko's MATLAB summary."""
    row = metric_row(windows, record_name, split)
    row["aggregation"] = "record_equal_errors; pooled_association_and_agreement"
    row["temko_report_label"] = temko_report_label
    row["n_records"] = int(len(per_record))
    row["pooled_window_mae_bpm"] = row["mae_bpm"]
    row["pooled_window_std_abs_error_bpm"] = row["std_abs_error_bpm"]
    row["pooled_window_mean_relative_error_percent"] = row["mean_relative_error_percent"]

    if not per_record.empty:
        row["mae_bpm"] = float(per_record["mae_bpm"].mean())
        row["std_abs_error_bpm"] = float(per_record["std_abs_error_bpm"].mean())
        row["mean_relative_error_percent"] = float(per_record["mean_relative_error_percent"].mean())
    return row


def build_metrics(windows: pd.DataFrame) -> pd.DataFrame:
    per_record_rows = []
    for record_name, record_df in windows.groupby("record", sort=False):
        split = str(record_df["split"].iloc[0])
        row = metric_row(record_df, record_name, split)
        row["aggregation"] = "per_record_windows"
        row["temko_report_label"] = ""
        row["n_records"] = 1
        row["pooled_window_mae_bpm"] = row["mae_bpm"]
        row["pooled_window_std_abs_error_bpm"] = row["std_abs_error_bpm"]
        row["pooled_window_mean_relative_error_percent"] = row["mean_relative_error_percent"]
        per_record_rows.append(row)

    per_record = pd.DataFrame(per_record_rows)
    rows = list(per_record_rows)
    for split, split_df in windows.groupby("split", sort=False):
        split_records = per_record[per_record["split"] == split]
        if split == "training" and len(split_records) == 12:
            report_label = "Err12"
        else:
            report_label = f"Err{len(split_records)}_{split}_available"
        rows.append(record_equal_summary(split_df, split_records, f"ALL_{split}", split, report_label))
    rows.append(
        record_equal_summary(
            windows,
            per_record,
            "ALL_DATABASE",
            "all",
            f"ErrAll{len(per_record)}_available",
        )
    )
    return pd.DataFrame(rows)


def plot_time_series(windows: pd.DataFrame, record_name: str, outdir: Path) -> Path:
    df = windows[windows["record"] == record_name]
    fig, ax = plt.subplots(figsize=(11.5, 4.7), constrained_layout=True)
    ax.plot(df["center_time_s"] / 60.0, df["hr_true_bpm"], label="BPM0 true HR", linewidth=1.8, color="#1f77b4")
    ax.plot(
        df["center_time_s"] / 60.0,
        df["temko_wfpv_vd_hr_bpm"],
        label="Temko WFPV-VD offline",
        linewidth=1.35,
        color="#9467bd",
    )
    ax.set_title(f"Temko WFPV-VD offline HR vs BPM0: {record_name}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Heart rate (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / f"{record_name}_timeseries_temko_wfpv_vd.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mae(metrics: pd.DataFrame, outdir: Path) -> Path:
    per_record = metrics[~metrics["record"].str.startswith("ALL_")].copy()
    per_record = per_record.sort_values(["split", "record"])
    colors = per_record["split"].map({"training": "#1f77b4", "competition": "#ff7f0e"}).fillna("#777777")
    fig, ax = plt.subplots(figsize=(13.0, 5.2), constrained_layout=True)
    ax.bar(per_record["record"], per_record["mae_bpm"], color=colors)
    ax.set_title("Temko WFPV-VD offline MAE by DATABASE record")
    ax.set_ylabel("MAE (bpm)")
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    path = outdir / "mae_by_record_temko_wfpv_vd.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter(windows: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 5.6), constrained_layout=True)
    for split, color in [("training", "#1f77b4"), ("competition", "#ff7f0e")]:
        df = windows[windows["split"] == split]
        ax.scatter(df["hr_true_bpm"], df["temko_wfpv_vd_hr_bpm"], s=12, alpha=0.35, label=split, color=color)
    vals = np.concatenate([windows["hr_true_bpm"].to_numpy(), windows["temko_wfpv_vd_hr_bpm"].to_numpy()])
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
        f"ALL: r={overall['pearson_r']:.3f}\\nrecord-equal MAE={overall['mae_bpm']:.2f} bpm",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.set_title("Temko WFPV-VD offline correlation")
    ax.set_xlabel("HR true BPM0 (bpm)")
    ax.set_ylabel("HR estimated (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "scatter_temko_wfpv_vd.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_bland_altman(windows: pd.DataFrame, metrics: pd.DataFrame, outdir: Path) -> Path:
    valid = windows[["hr_true_bpm", "temko_wfpv_vd_hr_bpm"]].dropna()
    mean_hr = valid.mean(axis=1)
    diff = valid["temko_wfpv_vd_hr_bpm"] - valid["hr_true_bpm"]
    overall = metrics[metrics["record"] == "ALL_DATABASE"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    ax.scatter(mean_hr, diff, s=12, alpha=0.35, color="#9467bd")
    bias = float(overall["bias_bpm"])
    lower = float(overall["bland_altman_lower_bpm"])
    upper = float(overall["bland_altman_upper_bpm"])
    ax.axhline(bias, color="#9467bd", linewidth=1.2, label="Bias")
    ax.axhline(lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
    ax.axhline(upper, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Bland-Altman: Temko WFPV-VD offline")
    ax.set_xlabel("Mean HR (bpm)")
    ax.set_ylabel("HR est - HR true (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = outdir / "bland_altman_temko_wfpv_vd.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Temko WFPV + Viterbi decoding on DATABASE records.")
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
    parser.add_argument("--plot-records", nargs="+", default=["DATA_01_TYPE01", "DATA_10_TYPE02", "TEST_S01_T01", "TEST_S07_T02"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.data_dir, args.records)
    if len(records) < 2:
        raise ValueError("Offline leave-one-record-out Viterbi decoding requires at least two records")
    if len(records) != 23 or "DATA_S04_T01" not in {record.name for record in records}:
        print(
            "WARNING: Temko's reference experiment uses 23 records including DATA_S04_T01; "
            f"this run has {len(records)} records, so each transition matrix is trained from "
            f"{len(records) - 1} available records."
        )
    emissions_by_record: dict[str, WfpvEmission] = {}
    for record in records:
        sig, bpm0 = load_record(record)
        emissions_by_record[record.name] = compute_wfpv_emission(
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

    windows = estimate_offline_vd(emissions_by_record)
    metrics = build_metrics(windows)

    windows_path = args.outdir / "temko_wfpv_vd_database_windows.csv"
    metrics_path = args.outdir / "temko_wfpv_vd_database_metrics.csv"
    matlab_summary_path = args.outdir / "temko_wfpv_vd_matlab_style_summary.csv"
    config_path = args.outdir / "temko_wfpv_vd_database_config.csv"
    windows.to_csv(windows_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["record"].str.startswith("ALL_")][
        [
            "temko_report_label",
            "record",
            "split",
            "n_records",
            "n_windows",
            "mae_bpm",
            "std_abs_error_bpm",
            "mean_relative_error_percent",
            "pooled_window_mae_bpm",
            "pearson_r",
            "bias_bpm",
            "bland_altman_lower_bpm",
            "bland_altman_upper_bpm",
        ]
    ].to_csv(matlab_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "source_repository": "https://github.com/andtem2000/PPG",
                "source_matlab_file": "PPG_WFPV_VD_TBME2017_offline.m",
                "transition_source": "leave-one-record-out BPM0 traces from the available DATABASE records",
                "data_dir": str(args.data_dir),
                "records": ",".join(records_by_name for records_by_name in emissions_by_record),
                "n_records_available": len(emissions_by_record),
                "transition_training_records_per_held_out": len(emissions_by_record) - 1,
                "reference_matlab_record_count": 23,
                "reference_transition_training_records_per_held_out": 22,
                "missing_reference_record": "DATA_S04_T01",
                "fs_hz": args.fs,
                "window_sec": args.window_sec,
                "step_sec": args.step_sec,
                "target_fs_hz": args.fs / 5.0,
                "n_fft": args.n_fft,
                "search_low_hz": args.search_low_hz,
                "search_high_hz": args.search_high_hz,
                "wf_length": args.wf_length,
                "evaluation_hr": "temko_wfpv_vd_hr_bpm, after MATLAB moving(...,4) smoothing",
                "error_summary_aggregation": "record_equal_as_in_Temko_MATLAB",
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
    print(f"Wrote MATLAB-style summary: {matlab_summary_path}")
    print(f"Wrote config: {config_path}")
    print("Plots:")
    for path in plot_paths:
        print(f"  {path}")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
