from __future__ import annotations

import json
from pathlib import Path


TARGET = Path("/Users/xiongzaizai/PPG/PPG_denosing.ipynb")


def md(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# PPG Baseline Denoising and HR Evaluation on DATABASE

本 notebook 使用 `DATABASE` 数据集和基线脚本 `ppg_fft_hr_analysis.py` 完成三件事：

1. 给出 `avAE`、`avRE`、`sdAE` 指标表。
2. 对比未滤波的 PPG 信号和基线预处理后的 PPG 信号。
3. 绘制 PPG FFT 估计心率与 `BPM0` 黄金心率的时间序列图、correlation 图和 Bland-Altman 图。

这里的基线预处理定义为：PPG1/PPG2 分别进行 0.4--4 Hz 四阶 Butterworth 带通滤波和 z-score 归一化，然后取平均。
""",
        "intro",
    ),
    code(
        """
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

WORKSPACE = Path("/Users/xiongzaizai/Documents/PPG")
SCRIPTS = WORKSPACE / "scripts"
DATA_DIR = Path("/Users/xiongzaizai/PPG/DATABASE")
OUTDIR = WORKSPACE / "outputs_ppg_denosing_notebook"
PLOTS_DIR = OUTDIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS))

print("DATABASE:", DATA_DIR, "exists =", DATA_DIR.exists())
print("Output:", OUTDIR)
""",
        "setup",
    ),
    md(
        """
## 1. 运行 DATABASE 版 PPG FFT baseline

该单元调用 `scripts/ppg_fft_hr_analysis.py`。脚本已经改为默认读取 `DATABASE` 数据集，并使用 `BPM0` 作为 ground truth。
""",
        "run-baseline-md",
    ),
    code(
        """
cmd = [
    sys.executable,
    str(SCRIPTS / "ppg_fft_hr_analysis.py"),
    "--data-dir", str(DATA_DIR),
    "--records", "all",
    "--outdir", str(OUTDIR),
    "--window-sec", "8",
    "--step-sec", "2",
    "--n-fft", "4096",
    "--search-low-hz", "0.8",
    "--search-high-hz", "3.0",
    "--ppg-low-hz", "0.4",
    "--ppg-high-hz", "4.0",
    "--smooth-windows", "5",
    "--plot-records", "DATA_01_TYPE01", "DATA_08_TYPE02", "TEST_S01_T01", "TEST_S07_T02",
]

env = os.environ.copy()
env["MPLCONFIGDIR"] = str(WORKSPACE / ".mplconfig")
result = subprocess.run(cmd, cwd=str(WORKSPACE), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-5000:])
if result.returncode != 0:
    raise RuntimeError(f"Baseline script failed with exit code {result.returncode}")
""",
        "run-baseline-code",
    ),
    md(
        """
## 2. avAE, avRE, sdAE 指标表

指标定义：

- `avAE`: average absolute error, 即平均绝对误差，单位 bpm。
- `avRE`: average relative absolute error, 即平均相对绝对误差，单位 %。
- `sdAE`: standard deviation of absolute error, 即绝对误差标准差，单位 bpm。
""",
        "metrics-md",
    ),
    code(
        """
windows = pd.read_csv(OUTDIR / "ppg_fft_hr_windows.csv")
metrics = pd.read_csv(OUTDIR / "ppg_fft_metrics.csv")

metric_table = metrics.loc[
    metrics["record"].str.startswith("ALL_"),
    ["record", "split", "method", "n_valid", "mae_bpm", "mean_relative_error_percent", "std_abs_error_bpm"],
].copy()

metric_table = metric_table.rename(columns={
    "mae_bpm": "avAE_bpm",
    "mean_relative_error_percent": "avRE_percent",
    "std_abs_error_bpm": "sdAE_bpm",
})
metric_table = metric_table.round({
    "avAE_bpm": 3,
    "avRE_percent": 3,
    "sdAE_bpm": 3,
})

metric_table.to_csv(OUTDIR / "ppg_fft_avAE_avRE_sdAE_table.csv", index=False)
display(metric_table)
print("Saved:", OUTDIR / "ppg_fft_avAE_avRE_sdAE_table.csv")
""",
        "metrics-code",
    ),
    md(
        """
## 3. 未处理 PPG 与基线预处理 PPG 对比

这里选取一个代表性窗口进行时域信号对比。上图为原始 PPG1/PPG2 平均信号；下图为同一窗口经过基线预处理后的 PPG 信号。
""",
        "raw-processed-md",
    ),
    code(
        """
from ppg_fft_hr_analysis import (
    discover_records,
    load_record,
    extract_ppg_channels,
    preprocess_ppg_window,
    zscore,
)

DEMO_RECORD = "DATA_10_TYPE02"
DEMO_WINDOW_INDEX = 111
FS = 125.0
WINDOW_SEC = 8.0
STEP_SEC = 2.0

record_map = {record.name: record for record in discover_records(DATA_DIR, ["all"])}
record = record_map[DEMO_RECORD]
sig, bpm0 = load_record(record)
ppg1, ppg2 = extract_ppg_channels(record, sig)

window_pts = int(round(WINDOW_SEC * FS))
step_pts = int(round(STEP_SEC * FS))
start = (DEMO_WINDOW_INDEX - 1) * step_pts
stop = start + window_pts
t = np.arange(window_pts) / FS

raw_ppg = 0.5 * ppg1[start:stop] + 0.5 * ppg2[start:stop]
ppg1_processed = preprocess_ppg_window(ppg1[start:stop], FS, low_hz=0.4, high_hz=4.0, order=4)
ppg2_processed = preprocess_ppg_window(ppg2[start:stop], FS, low_hz=0.4, high_hz=4.0, order=4)
processed_ppg = 0.5 * ppg1_processed + 0.5 * ppg2_processed

fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.8), sharex=True, constrained_layout=True)
axes[0].plot(t, raw_ppg, color="#4c78a8", linewidth=1.0)
axes[0].set_title(f"Raw PPG mean signal: {DEMO_RECORD}, window {DEMO_WINDOW_INDEX}")
axes[0].set_ylabel("Raw amplitude")
axes[0].grid(True, alpha=0.25)

axes[1].plot(t, processed_ppg, color="#d62728", linewidth=1.0)
axes[1].set_title("Baseline-processed PPG: 0.4-4 Hz band-pass + z-score")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Normalized amplitude")
axes[1].grid(True, alpha=0.25)

raw_processed_path = PLOTS_DIR / f"{DEMO_RECORD}_raw_vs_baseline_processed_ppg.png"
fig.savefig(raw_processed_path, dpi=220)
plt.close(fig)

display(Image(filename=str(raw_processed_path), width=900))
print("Saved:", raw_processed_path)
""",
        "raw-processed-code",
    ),
    md(
        """
## 4. 与 BPM0 的时间序列对比

该图展示每个 8 s 窗口中 `BPM0` 与 PPG FFT 估计心率的变化。
""",
        "timeseries-md",
    ),
    code(
        """
PLOT_RECORD = "DATA_10_TYPE02"
record_df = windows[windows["record"] == PLOT_RECORD].copy()

fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
ax.plot(record_df["center_time_s"] / 60.0, record_df["hr_true_bpm"], label="BPM0 true HR", color="#1f77b4", linewidth=1.7)
ax.plot(record_df["center_time_s"] / 60.0, record_df["ppg_fft_hr_est_bpm"], label="PPG FFT HR", color="#d62728", linewidth=1.2)
ax.set_title(f"PPG FFT HR vs BPM0: {PLOT_RECORD}")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heart rate (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

timeseries_path = PLOTS_DIR / f"{PLOT_RECORD}_hr_true_vs_ppg_fft.png"
fig.savefig(timeseries_path, dpi=220)
plt.close(fig)

display(Image(filename=str(timeseries_path), width=950))
print("Saved:", timeseries_path)
""",
        "timeseries-code",
    ),
    md(
        """
## 5. Correlation 图

横轴为 `BPM0` 黄金心率，纵轴为 PPG FFT 估计心率。黑色对角线表示理想估计。
""",
        "correlation-md",
    ),
    code(
        """
valid = windows[["hr_true_bpm", "ppg_fft_hr_est_bpm"]].dropna()
overall = metrics[(metrics["record"] == "ALL_DATABASE") & (metrics["method"] == "PPG FFT")].iloc[0]

fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
ax.scatter(valid["hr_true_bpm"], valid["ppg_fft_hr_est_bpm"], s=13, alpha=0.35, color="#1f77b4")
lo = float(np.nanmin(valid.to_numpy()) - 5)
hi = float(np.nanmax(valid.to_numpy()) + 5)
ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_title("Correlation: PPG FFT HR vs BPM0")
ax.set_xlabel("BPM0 true HR (bpm)")
ax.set_ylabel("PPG FFT HR (bpm)")
ax.grid(True, alpha=0.25)
ax.text(
    0.04,
    0.96,
    f"r={overall['pearson_r']:.3f}\\navAE={overall['mae_bpm']:.2f} bpm",
    transform=ax.transAxes,
    va="top",
    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

corr_path = PLOTS_DIR / "correlation_ppg_fft_vs_bpm0.png"
fig.savefig(corr_path, dpi=220)
plt.close(fig)

display(Image(filename=str(corr_path), width=720))
print("Saved:", corr_path)
""",
        "correlation-code",
    ),
    md(
        """
## 6. Bland-Altman 图

横轴为估计心率和真实心率的均值，纵轴为估计误差 `HR_est - BPM0`。红线表示 bias，黑色虚线表示 95% limits of agreement。
""",
        "bland-md",
    ),
    code(
        """
mean_hr = valid.mean(axis=1)
diff_hr = valid["ppg_fft_hr_est_bpm"] - valid["hr_true_bpm"]
bias = float(overall["bias_bpm"])
lower = float(overall["bland_altman_lower_bpm"])
upper = float(overall["bland_altman_upper_bpm"])

fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
ax.scatter(mean_hr, diff_hr, s=13, alpha=0.35, color="#1f77b4")
ax.axhline(bias, color="#d62728", linewidth=1.2, label=f"Bias = {bias:.2f} bpm")
ax.axhline(lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
ax.axhline(upper, color="black", linestyle="--", linewidth=1.0)
ax.set_title("Bland-Altman: PPG FFT HR vs BPM0")
ax.set_xlabel("Mean HR (bpm)")
ax.set_ylabel("HR_est - BPM0 (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

ba_path = PLOTS_DIR / "bland_altman_ppg_fft_vs_bpm0.png"
fig.savefig(ba_path, dpi=220)
plt.close(fig)

display(Image(filename=str(ba_path), width=760))
print("Saved:", ba_path)
""",
        "bland-code",
    ),
    md(
        """
## 7. 输出文件
""",
        "outputs-md",
    ),
    code(
        """
print("Metric table:", OUTDIR / "ppg_fft_avAE_avRE_sdAE_table.csv")
print("Window-level HR:", OUTDIR / "ppg_fft_hr_windows.csv")
print("Baseline metrics:", OUTDIR / "ppg_fft_metrics.csv")
print("Plots directory:", PLOTS_DIR)
""",
        "outputs-code",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

TARGET.parent.mkdir(parents=True, exist_ok=True)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {TARGET} with {len(cells)} cells")
