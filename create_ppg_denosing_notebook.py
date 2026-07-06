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
# PPG FFT and ACC-NLMS Denoising HR Evaluation on DATABASE

本 notebook 使用 `DATABASE` 数据集，对比 PPG FFT baseline 和 ACC-NLMS FFT 两种方法，完成以下内容：

1. 给出 PPG FFT 与 ACC-NLMS FFT 的 `avAE`、`avRE`、`sdAE` 指标表。
2. 对比未滤波的 PPG 信号和基线预处理后的 PPG 信号。
3. 绘制 PPG FFT 估计心率与 `BPM0` 黄金心率的时间序列图、correlation 图和 Bland-Altman 图。
4. 绘制 ACC-NLMS FFT 单独的 cleaned PPG、时间序列图、correlation 图和 Bland-Altman 图。
5. 绘制 PPG FFT 与 ACC-NLMS FFT 的两方法对比图。

这里的基线预处理定义为：PPG1/PPG2 分别进行 0.4--4 Hz 四阶 Butterworth 带通滤波和 z-score 归一化，然后取平均。
ACC-NLMS 方法使用 ACC 三轴作为参考信号估计运动伪影，并以 NLMS 输出误差作为 cleaned PPG，再进行同样的窗口级 FFT 心率估计。
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
## 7. 运行 ACC-NLMS FFT

该单元调用 `scripts/ppg_nlms_acc_fft_hr_analysis.py`。它使用 ACC 三轴作为参考信号，通过 NLMS 自适应滤波获得 cleaned PPG，再用同样的 8 s 窗口、2 s 步长和 FFT 主峰法估计心率。
""",
        "run-acc-nlms-md",
    ),
    code(
        """
cmd_nlms = [
    sys.executable,
    str(SCRIPTS / "ppg_nlms_acc_fft_hr_analysis.py"),
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
    "--nlms-filter-order", "32",
    "--nlms-mu", "0.005",
    "--smooth-windows", "5",
    "--plot-records", "DATA_01_TYPE01", "DATA_08_TYPE02", "DATA_10_TYPE02", "TEST_S01_T01", "TEST_S07_T02",
]

result = subprocess.run(cmd_nlms, cwd=str(WORKSPACE), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-6000:])
if result.returncode != 0:
    raise RuntimeError(f"ACC-NLMS FFT script failed with exit code {result.returncode}")
""",
        "run-acc-nlms-code",
    ),
    md(
        """
## 8. ACC-NLMS FFT 的 avAE, avRE, sdAE 指标表
""",
        "acc-nlms-metrics-md",
    ),
    code(
        """
nlms_windows = pd.read_csv(OUTDIR / "ppg_nlms_acc_fft_hr_windows.csv")
nlms_metrics = pd.read_csv(OUTDIR / "ppg_nlms_acc_fft_metrics.csv")

acc_metric_table = nlms_metrics.loc[
    (nlms_metrics["record"].str.startswith("ALL_")) & (nlms_metrics["method"] == "ACC-NLMS FFT"),
    ["record", "split", "method", "n_valid", "mae_bpm", "mean_relative_error_percent", "std_abs_error_bpm", "pearson_r"],
].copy()

acc_metric_table = acc_metric_table.rename(columns={
    "mae_bpm": "avAE_bpm",
    "mean_relative_error_percent": "avRE_percent",
    "std_abs_error_bpm": "sdAE_bpm",
})
acc_metric_table = acc_metric_table.round({
    "avAE_bpm": 3,
    "avRE_percent": 3,
    "sdAE_bpm": 3,
    "pearson_r": 3,
})

acc_metric_table.to_csv(OUTDIR / "acc_nlms_fft_avAE_avRE_sdAE_table.csv", index=False)
display(acc_metric_table)
print("Saved:", OUTDIR / "acc_nlms_fft_avAE_avRE_sdAE_table.csv")
""",
        "acc-nlms-metrics-code",
    ),
    md(
        """
## 9. ACC-NLMS-only: 输入 PPG、cleaned PPG 与 artifact 对比

该图对应 FFT 部分的“未处理 PPG 与预处理 PPG 对比”。这里额外展示 NLMS 估计出的与 ACC 相关的 artifact，便于说明 ACC-NLMS 的去伪影作用。
""",
        "acc-nlms-signal-md",
    ),
    code(
        """
from ppg_nlms_acc_fft_hr_analysis import make_nlms_cleaned_ppg

desired, cleaned, artifact = make_nlms_cleaned_ppg(
    record,
    sig,
    fs=FS,
    ppg_low_hz=0.4,
    ppg_high_hz=4.0,
    filter_order=4,
    nlms_filter_order=32,
    nlms_mu=0.005,
    nlms_eps=1e-6,
)

fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.2), sharex=True, constrained_layout=True)
axes[0].plot(t, raw_ppg, color="#4c78a8", linewidth=1.0)
axes[0].set_title(f"Raw PPG mean signal: {DEMO_RECORD}, window {DEMO_WINDOW_INDEX}")
axes[0].set_ylabel("Raw amp.")

axes[1].plot(t, cleaned[start:stop], color="#d62728", linewidth=1.0)
axes[1].set_title("ACC-NLMS cleaned PPG")
axes[1].set_ylabel("Cleaned amp.")

axes[2].plot(t, artifact[start:stop], color="#7f7f7f", linewidth=1.0)
axes[2].set_title("NLMS-estimated ACC-correlated artifact")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Artifact")

for ax in axes:
    ax.grid(True, alpha=0.25)

acc_signal_path = PLOTS_DIR / f"{DEMO_RECORD}_raw_vs_acc_nlms_cleaned_ppg.png"
fig.savefig(acc_signal_path, dpi=220)
plt.close(fig)

display(Image(filename=str(acc_signal_path), width=900))
print("Saved:", acc_signal_path)
""",
        "acc-nlms-signal-code",
    ),
    md(
        """
## 10. ACC-NLMS-only: 与 BPM0 的时间序列对比
""",
        "acc-nlms-timeseries-md",
    ),
    code(
        """
acc_record_df = nlms_windows[nlms_windows["record"] == PLOT_RECORD].copy()

fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
ax.plot(acc_record_df["center_time_s"] / 60.0, acc_record_df["hr_true_bpm"], label="BPM0 true HR", color="#1f77b4", linewidth=1.7)
ax.plot(acc_record_df["center_time_s"] / 60.0, acc_record_df["ppg_acc_nlms_fft_hr_est_bpm"], label="ACC-NLMS FFT HR", color="#d62728", linewidth=1.2)
ax.set_title(f"ACC-NLMS FFT HR vs BPM0: {PLOT_RECORD}")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heart rate (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

acc_timeseries_path = PLOTS_DIR / f"{PLOT_RECORD}_hr_true_vs_acc_nlms_fft.png"
fig.savefig(acc_timeseries_path, dpi=220)
plt.close(fig)

display(Image(filename=str(acc_timeseries_path), width=950))
print("Saved:", acc_timeseries_path)
""",
        "acc-nlms-timeseries-code",
    ),
    md(
        """
## 11. ACC-NLMS-only: Correlation 图
""",
        "acc-nlms-correlation-md",
    ),
    code(
        """
acc_valid = nlms_windows[["hr_true_bpm", "ppg_acc_nlms_fft_hr_est_bpm"]].dropna()
acc_overall = nlms_metrics[
    (nlms_metrics["record"] == "ALL_DATABASE") & (nlms_metrics["method"] == "ACC-NLMS FFT")
].iloc[0]

fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
ax.scatter(acc_valid["hr_true_bpm"], acc_valid["ppg_acc_nlms_fft_hr_est_bpm"], s=13, alpha=0.35, color="#d62728")
lo = float(np.nanmin(acc_valid.to_numpy()) - 5)
hi = float(np.nanmax(acc_valid.to_numpy()) + 5)
ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_title("Correlation: ACC-NLMS FFT HR vs BPM0")
ax.set_xlabel("BPM0 true HR (bpm)")
ax.set_ylabel("ACC-NLMS FFT HR (bpm)")
ax.grid(True, alpha=0.25)
ax.text(
    0.04,
    0.96,
    f"r={acc_overall['pearson_r']:.3f}\\navAE={acc_overall['mae_bpm']:.2f} bpm",
    transform=ax.transAxes,
    va="top",
    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
)

acc_corr_path = PLOTS_DIR / "correlation_acc_nlms_fft_vs_bpm0.png"
fig.savefig(acc_corr_path, dpi=220)
plt.close(fig)

display(Image(filename=str(acc_corr_path), width=720))
print("Saved:", acc_corr_path)
""",
        "acc-nlms-correlation-code",
    ),
    md(
        """
## 12. ACC-NLMS-only: Bland-Altman 图
""",
        "acc-nlms-bland-md",
    ),
    code(
        """
acc_mean_hr = acc_valid.mean(axis=1)
acc_diff_hr = acc_valid["ppg_acc_nlms_fft_hr_est_bpm"] - acc_valid["hr_true_bpm"]
acc_bias = float(acc_overall["bias_bpm"])
acc_lower = float(acc_overall["bland_altman_lower_bpm"])
acc_upper = float(acc_overall["bland_altman_upper_bpm"])

fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
ax.scatter(acc_mean_hr, acc_diff_hr, s=13, alpha=0.35, color="#d62728")
ax.axhline(acc_bias, color="#d62728", linewidth=1.2, label=f"Bias = {acc_bias:.2f} bpm")
ax.axhline(acc_lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
ax.axhline(acc_upper, color="black", linestyle="--", linewidth=1.0)
ax.set_title("Bland-Altman: ACC-NLMS FFT HR vs BPM0")
ax.set_xlabel("Mean HR (bpm)")
ax.set_ylabel("HR_est - BPM0 (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)

acc_ba_path = PLOTS_DIR / "bland_altman_acc_nlms_fft_vs_bpm0.png"
fig.savefig(acc_ba_path, dpi=220)
plt.close(fig)

display(Image(filename=str(acc_ba_path), width=760))
print("Saved:", acc_ba_path)
""",
        "acc-nlms-bland-code",
    ),
    md(
        """
## 13. PPG FFT 与 ACC-NLMS FFT 指标对比表
""",
        "comparison-table-md",
    ),
    code(
        """
fft_summary = metrics.loc[
    metrics["record"].str.startswith("ALL_"),
    ["record", "split", "method", "n_valid", "mae_bpm", "mean_relative_error_percent", "std_abs_error_bpm", "pearson_r"],
].copy()
acc_summary = nlms_metrics.loc[
    (nlms_metrics["record"].str.startswith("ALL_")) & (nlms_metrics["method"] == "ACC-NLMS FFT"),
    ["record", "split", "method", "n_valid", "mae_bpm", "mean_relative_error_percent", "std_abs_error_bpm", "pearson_r"],
].copy()

two_method_table = pd.concat([fft_summary, acc_summary], ignore_index=True)
two_method_table = two_method_table.rename(columns={
    "mae_bpm": "avAE_bpm",
    "mean_relative_error_percent": "avRE_percent",
    "std_abs_error_bpm": "sdAE_bpm",
})
two_method_table = two_method_table.round({
    "avAE_bpm": 3,
    "avRE_percent": 3,
    "sdAE_bpm": 3,
    "pearson_r": 3,
})
two_method_table.to_csv(OUTDIR / "ppg_fft_vs_acc_nlms_avAE_avRE_sdAE_table.csv", index=False)
display(two_method_table)
print("Saved:", OUTDIR / "ppg_fft_vs_acc_nlms_avAE_avRE_sdAE_table.csv")
""",
        "comparison-table-code",
    ),
    md(
        """
## 14. 两种方法对比图

下面依次给出 PPG FFT 与 ACC-NLMS FFT 的时间序列对比、avAE 柱状图、correlation 对比和 Bland-Altman 对比。
""",
        "comparison-plots-md",
    ),
    code(
        """
two_method_windows = windows[[
    "record", "split", "window_index", "center_time_s", "hr_true_bpm", "ppg_fft_hr_est_bpm"
]].merge(
    nlms_windows[["record", "center_time_s", "ppg_acc_nlms_fft_hr_est_bpm"]],
    on=["record", "center_time_s"],
    how="inner",
)
two_method_windows.to_csv(OUTDIR / "ppg_fft_vs_acc_nlms_window_level_results.csv", index=False)

plot_df = two_method_windows[two_method_windows["record"] == PLOT_RECORD].copy()
fig, ax = plt.subplots(figsize=(12.2, 4.9), constrained_layout=True)
ax.plot(plot_df["center_time_s"] / 60.0, plot_df["hr_true_bpm"], label="BPM0 true HR", color="#1f77b4", linewidth=1.8)
ax.plot(plot_df["center_time_s"] / 60.0, plot_df["ppg_fft_hr_est_bpm"], label="PPG FFT", color="#ff7f0e", linewidth=1.1)
ax.plot(plot_df["center_time_s"] / 60.0, plot_df["ppg_acc_nlms_fft_hr_est_bpm"], label="ACC-NLMS FFT", color="#d62728", linewidth=1.2)
ax.set_title(f"PPG FFT vs ACC-NLMS FFT HR estimation: {PLOT_RECORD}")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heart rate (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, ncol=3)

two_ts_path = PLOTS_DIR / f"{PLOT_RECORD}_ppg_fft_vs_acc_nlms_timeseries.png"
fig.savefig(two_ts_path, dpi=220)
plt.close(fig)
display(Image(filename=str(two_ts_path), width=950))
print("Saved:", two_ts_path)

records_for_plot = ["ALL_training", "ALL_competition", "ALL_DATABASE"]
labels_for_plot = ["Training", "Competition", "All"]
method_colors = {"PPG FFT": "#0072B2", "ACC-NLMS FFT": "#D55E00"}
x = np.arange(len(records_for_plot), dtype=float)
width = 0.34

fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
for i, method in enumerate(["PPG FFT", "ACC-NLMS FFT"]):
    values = []
    for record_name in records_for_plot:
        row = two_method_table[(two_method_table["record"] == record_name) & (two_method_table["method"] == method)]
        values.append(float(row["avAE_bpm"].iloc[0]) if not row.empty else np.nan)
    ax.bar(x + (i - 0.5) * width, values, width=width, color=method_colors[method], label=method)
ax.set_xticks(x)
ax.set_xticklabels(labels_for_plot)
ax.set_ylabel("avAE / MAE (bpm)")
ax.set_title("PPG FFT vs ACC-NLMS FFT avAE comparison")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)

two_mae_path = PLOTS_DIR / "mae_ppg_fft_vs_acc_nlms_fft.png"
fig.savefig(two_mae_path, dpi=220)
plt.close(fig)
display(Image(filename=str(two_mae_path), width=820))
print("Saved:", two_mae_path)
""",
        "comparison-timeseries-mae-code",
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
comparison_specs = [
    ("PPG FFT", "ppg_fft_hr_est_bpm", "#0072B2"),
    ("ACC-NLMS FFT", "ppg_acc_nlms_fft_hr_est_bpm", "#D55E00"),
]
vals = [two_method_windows["hr_true_bpm"].to_numpy()]
text_lines = []
for method, col, color in comparison_specs:
    cur = two_method_windows[["hr_true_bpm", col]].dropna()
    ax.scatter(cur["hr_true_bpm"], cur[col], s=13, alpha=0.36, label=method, color=color)
    vals.append(cur[col].to_numpy())
    row = two_method_table[(two_method_table["record"] == "ALL_DATABASE") & (two_method_table["method"] == method)].iloc[0]
    text_lines.append(f"{method}: r={row['pearson_r']:.3f}, avAE={row['avAE_bpm']:.2f}")
finite = np.concatenate([v[np.isfinite(v)] for v in vals if v.size])
lo = float(finite.min() - 5)
hi = float(finite.max() + 5)
ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.text(0.04, 0.96, "\\n".join(text_lines), transform=ax.transAxes, va="top", fontsize=9)
ax.set_title("Correlation: PPG FFT vs ACC-NLMS FFT")
ax.set_xlabel("BPM0 true HR (bpm)")
ax.set_ylabel("Estimated HR (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="lower right")

two_corr_path = PLOTS_DIR / "correlation_ppg_fft_vs_acc_nlms_fft.png"
fig.savefig(two_corr_path, dpi=220)
plt.close(fig)
display(Image(filename=str(two_corr_path), width=740))
print("Saved:", two_corr_path)

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)
for ax, (method, col, color) in zip(axes, comparison_specs):
    cur = two_method_windows[["hr_true_bpm", col]].dropna()
    cur_mean = cur.mean(axis=1)
    cur_diff = cur[col] - cur["hr_true_bpm"]
    source_metrics = metrics if method == "PPG FFT" else nlms_metrics
    row = source_metrics[
        (source_metrics["record"] == "ALL_DATABASE") & (source_metrics["method"] == method)
    ].iloc[0]
    cur_bias = float(row["bias_bpm"])
    cur_lower = float(row["bland_altman_lower_bpm"])
    cur_upper = float(row["bland_altman_upper_bpm"])
    ax.scatter(cur_mean, cur_diff, s=11, alpha=0.32, color=color)
    ax.axhline(cur_bias, color=color, linewidth=1.2, label="Bias")
    ax.axhline(cur_lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
    ax.axhline(cur_upper, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(method)
    ax.set_xlabel("Mean HR (bpm)")
    ax.set_ylabel("HR_est - BPM0 (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

two_ba_path = PLOTS_DIR / "bland_altman_ppg_fft_vs_acc_nlms_fft.png"
fig.savefig(two_ba_path, dpi=220)
plt.close(fig)
display(Image(filename=str(two_ba_path), width=1000))
print("Saved:", two_ba_path)
""",
        "comparison-corr-ba-code",
    ),
    md(
        """
## 15. 输出文件
""",
        "outputs-md",
    ),
    code(
        """
print("PPG FFT metric table:", OUTDIR / "ppg_fft_avAE_avRE_sdAE_table.csv")
print("ACC-NLMS metric table:", OUTDIR / "acc_nlms_fft_avAE_avRE_sdAE_table.csv")
print("Two-method comparison table:", OUTDIR / "ppg_fft_vs_acc_nlms_avAE_avRE_sdAE_table.csv")
print("PPG FFT window-level HR:", OUTDIR / "ppg_fft_hr_windows.csv")
print("ACC-NLMS window-level HR:", OUTDIR / "ppg_nlms_acc_fft_hr_windows.csv")
print("Two-method window-level results:", OUTDIR / "ppg_fft_vs_acc_nlms_window_level_results.csv")
print("Baseline metrics:", OUTDIR / "ppg_fft_metrics.csv")
print("ACC-NLMS metrics:", OUTDIR / "ppg_nlms_acc_fft_metrics.csv")
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
