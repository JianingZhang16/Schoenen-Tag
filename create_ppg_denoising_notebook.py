from __future__ import annotations

import json
from pathlib import Path


TARGET = Path("/Users/xiongzaizai/PPG/PPG_denoising.ipynb")


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
# PPG Denoising on DATABASE: PPG FFT, ACC-NLMS FFT and Temko WFPV

本 notebook 使用 `DATABASE` 数据集，对比四种方法：

1. **PPG FFT baseline**：PPG1/PPG2 经 0.4--4 Hz 带通滤波、z-score 归一化、平均后，用 FFT 主峰估计心率。
2. **ACC-NLMS FFT**：使用 ACC 三轴作为参考信号，通过 NLMS 自适应滤波去除与运动相关的伪影，再用 FFT 估计心率。
3. **Temko WFPV**：复现 Temko 论文和 Matlab 代码 `PPG_WFPV_TBME2017.m` 中的 online WFPV 方法，即 Wiener filter + phase vocoder + history-constrained tracking。
4. **Temko WFPV-VD offline**：复现 `PPG_WFPV_VD_TBME2017_offline.m` 中的离线 Viterbi decoding 版本，用作非实时离线上限参考。

输出内容：

- `avAE`、`avRE`、`sdAE` 指标表；
- Temko WFPV online 与 WFPV-VD offline 单独的指标表和图；
- PPG FFT、ACC-NLMS FFT、Temko WFPV online、Temko WFPV-VD offline 的终极对比表；
- 原始 PPG、baseline 预处理 PPG、ACC-NLMS cleaned PPG 和 NLMS artifact 图；
- 四种方法与 `BPM0` 黄金心率的时间序列图、correlation 图和 Bland-Altman 图。
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
PPG_MASTER = Path("/Users/xiongzaizai/Downloads/PPG-master")
OUTDIR = WORKSPACE / "outputs_ppg_denoising_notebook"
PLOTS_DIR = OUTDIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS))

print("DATABASE:", DATA_DIR, "exists =", DATA_DIR.exists())
print("Temko Matlab code:", PPG_MASTER / "PPG_WFPV_TBME2017.m")
print("Output:", OUTDIR)
""",
        "setup",
    ),
    md(
        """
## 1. Temko Matlab 代码与本文复现关系

Temko 论文公开代码中：

- `PPG_WFPV_TBME2017.m` 是 online HR estimation；
- `PPG_WFPV_VD_TBME2017_offline.m` 是 offline Viterbi decoding 版本。

本文复现两个 Temko 版本：

1. **online WFPV**，对应 `PPG_WFPV_TBME2017.m`；
2. **offline WFPV-VD**，对应 `PPG_WFPV_VD_TBME2017_offline.m`。

online WFPV 的主要步骤为：

`PPG/ACC band-pass filtering -> PPG channel averaging -> downsample to 25 Hz -> DFT -> Wiener spectral weighting -> phase vocoder frequency refinement -> history-constrained tracking -> BPM0 evaluation`

offline WFPV-VD 先计算整段记录的 WFPV 发射矩阵，再用其他记录的 `BPM0` 构建转移矩阵，并通过 Viterbi decoding 得到全局最优 HR 轨迹。它不是实时算法，但很适合作为 Temko 方法的离线性能上限参考。
""",
        "temko-method-md",
    ),
    md(
        """
## 2. 运行 PPG FFT baseline
""",
        "run-fft-md",
    ),
    code(
        """
env = os.environ.copy()
env["MPLCONFIGDIR"] = str(WORKSPACE / ".mplconfig")

cmd_fft = [
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
    "--plot-records", "DATA_01_TYPE01", "DATA_08_TYPE02", "DATA_10_TYPE02", "TEST_S01_T01", "TEST_S07_T02",
]
result = subprocess.run(cmd_fft, cwd=str(WORKSPACE), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-5000:])
if result.returncode != 0:
    raise RuntimeError(f"PPG FFT baseline failed with exit code {result.returncode}")
""",
        "run-fft-code",
    ),
    md(
        """
## 3. 运行 ACC-NLMS FFT
""",
        "run-nlms-md",
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
    raise RuntimeError(f"ACC-NLMS FFT failed with exit code {result.returncode}")
""",
        "run-nlms-code",
    ),
    md(
        """
## 4. 运行 Temko WFPV online 方法

该单元调用 Python 复现脚本 `temko_wfpv_database_analysis.py`，其流程对应 Matlab 文件 `PPG_WFPV_TBME2017.m`。
""",
        "run-temko-md",
    ),
    code(
        """
cmd_temko = [
    sys.executable,
    str(SCRIPTS / "temko_wfpv_database_analysis.py"),
    "--data-dir", str(DATA_DIR),
    "--records", "all",
    "--outdir", str(OUTDIR),
    "--window-sec", "8",
    "--step-sec", "2",
    "--n-fft", "1024",
    "--search-low-hz", "1.0",
    "--search-high-hz", "3.0",
    "--wf-length", "15",
    "--smooth-freq-bins", "3",
    "--plot-records", "DATA_01_TYPE01", "DATA_08_TYPE02", "DATA_10_TYPE02", "TEST_S01_T01", "TEST_S07_T02",
]
result = subprocess.run(cmd_temko, cwd=str(WORKSPACE), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-6000:])
if result.returncode != 0:
    raise RuntimeError(f"Temko WFPV failed with exit code {result.returncode}")
""",
        "run-temko-code",
    ),
    md(
        """
## 5. 运行 Temko WFPV-VD offline 方法

该单元调用 Python 复现脚本 `temko_wfpv_vd_database_analysis.py`，其流程对应 Matlab 文件 `PPG_WFPV_VD_TBME2017_offline.m`。该方法使用整段记录进行 Viterbi decoding，因此结果代表离线处理效果，不代表实时估计能力。
""",
        "run-temko-vd-md",
    ),
    code(
        """
cmd_temko_vd = [
    sys.executable,
    str(SCRIPTS / "temko_wfpv_vd_database_analysis.py"),
    "--data-dir", str(DATA_DIR),
    "--records", "all",
    "--outdir", str(OUTDIR),
    "--window-sec", "8",
    "--step-sec", "2",
    "--n-fft", "1024",
    "--search-low-hz", "1.0",
    "--search-high-hz", "3.0",
    "--wf-length", "15",
    "--smooth-freq-bins", "3",
    "--plot-records", "DATA_01_TYPE01", "DATA_08_TYPE02", "DATA_10_TYPE02", "TEST_S01_T01", "TEST_S03_T02", "TEST_S07_T02",
]
result = subprocess.run(cmd_temko_vd, cwd=str(WORKSPACE), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout[-6000:])
if result.returncode != 0:
    raise RuntimeError(f"Temko WFPV-VD offline failed with exit code {result.returncode}")
""",
        "run-temko-vd-code",
    ),
    md(
        """
## 6. avAE、avRE、sdAE 指标表

指标定义：

- `avAE`: average absolute error，平均绝对误差，单位 bpm；
- `avRE`: average relative absolute error，平均相对绝对误差，单位 %；
- `sdAE`: standard deviation of absolute error，绝对误差标准差，单位 bpm。
""",
        "metrics-md",
    ),
    code(
        """
fft_windows = pd.read_csv(OUTDIR / "ppg_fft_hr_windows.csv")
fft_metrics = pd.read_csv(OUTDIR / "ppg_fft_metrics.csv")
comparison_windows = pd.read_csv(OUTDIR / "ppg_nlms_acc_fft_hr_windows.csv")
comparison_metrics = pd.read_csv(OUTDIR / "ppg_nlms_acc_fft_metrics.csv")
temko_windows = pd.read_csv(OUTDIR / "temko_wfpv_database_windows.csv")
temko_metrics = pd.read_csv(OUTDIR / "temko_wfpv_database_metrics.csv")
temko_vd_windows = pd.read_csv(OUTDIR / "temko_wfpv_vd_database_windows.csv")
temko_vd_metrics = pd.read_csv(OUTDIR / "temko_wfpv_vd_database_metrics.csv")

temko_metrics = temko_metrics.copy()
temko_metrics.insert(2, "method", "Temko WFPV")
temko_vd_metrics = temko_vd_metrics.copy()
temko_vd_metrics.insert(2, "method", "Temko WFPV-VD offline")

all_metrics = pd.concat(
    [
        comparison_metrics,
        temko_metrics,
        temko_vd_metrics,
    ],
    ignore_index=True,
    sort=False,
)

metric_table = all_metrics.loc[
    all_metrics["record"].str.startswith("ALL_"),
    ["record", "split", "method", "n_valid", "mae_bpm", "mean_relative_error_percent", "std_abs_error_bpm", "pearson_r"],
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
    "pearson_r": 3,
})
metric_table.to_csv(OUTDIR / "four_methods_avAE_avRE_sdAE_table.csv", index=False)
display(metric_table)
print("Saved:", OUTDIR / "four_methods_avAE_avRE_sdAE_table.csv")
""",
        "metrics-code",
    ),
    md(
        """
## 7. 终极表格：四种方法效果对比

该表给出 PPG FFT、ACC-NLMS FFT、Temko WFPV online 和 Temko WFPV-VD offline 的直接对比，并标出每个数据划分下 `avAE` 最低的方法。
""",
        "ultimate-table-md",
    ),
    code(
        """
method_order = ["PPG FFT", "ACC-NLMS FFT", "Temko WFPV", "Temko WFPV-VD offline"]
rows = []
for record_name in ["ALL_training", "ALL_competition", "ALL_DATABASE"]:
    subset = metric_table[metric_table["record"] == record_name].copy()
    row = {
        "record": record_name,
        "split": subset["split"].iloc[0],
    }
    for method in method_order:
        m = subset[subset["method"] == method].iloc[0]
        key = method.replace("-", "_").replace(" ", "_")
        row[f"{key}_avAE_bpm"] = float(m["avAE_bpm"])
        row[f"{key}_avRE_percent"] = float(m["avRE_percent"])
        row[f"{key}_sdAE_bpm"] = float(m["sdAE_bpm"])
        row[f"{key}_pearson_r"] = float(m["pearson_r"])
    best = subset.loc[subset["avAE_bpm"].idxmin()]
    row["best_method_by_avAE"] = best["method"]
    row["best_avAE_bpm"] = float(best["avAE_bpm"])
    rows.append(row)

ultimate_table = pd.DataFrame(rows).round(3)
ultimate_table.to_csv(OUTDIR / "ultimate_four_method_comparison_table.csv", index=False)
display(ultimate_table)
print("Saved:", OUTDIR / "ultimate_four_method_comparison_table.csv")
""",
        "ultimate-table-code",
    ),
    md(
        """
## 8. 原始 PPG、baseline 预处理 PPG 与 ACC-NLMS 清理后 PPG

这里选取 `DATA_10_TYPE02` 的一个运动伪影较明显窗口，展示同一时间段中：

1. 未处理的 PPG1/PPG2 平均信号；
2. baseline 预处理后的 PPG；
3. ACC-NLMS 清理后的 PPG；
4. NLMS 估计出的 ACC-correlated artifact。
""",
        "signal-md",
    ),
    code(
        """
from ppg_fft_hr_analysis import (
    discover_records,
    load_record,
    extract_ppg_channels,
    preprocess_ppg_window,
)
from ppg_nlms_acc_fft_hr_analysis import make_nlms_cleaned_ppg

DEMO_RECORD = "DATA_10_TYPE02"
DEMO_WINDOW_INDEX = 111
FS = 125.0
WINDOW_SEC = 8.0
STEP_SEC = 2.0

record_map = {record.name: record for record in discover_records(DATA_DIR, ["all"])}
record = record_map[DEMO_RECORD]
sig, bpm0 = load_record(record)
ppg1, ppg2 = extract_ppg_channels(record, sig)

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

window_pts = int(round(WINDOW_SEC * FS))
step_pts = int(round(STEP_SEC * FS))
start = (DEMO_WINDOW_INDEX - 1) * step_pts
stop = start + window_pts
t = np.arange(window_pts) / FS

raw_ppg = 0.5 * ppg1[start:stop] + 0.5 * ppg2[start:stop]
ppg1_processed = preprocess_ppg_window(ppg1[start:stop], FS, low_hz=0.4, high_hz=4.0, order=4)
ppg2_processed = preprocess_ppg_window(ppg2[start:stop], FS, low_hz=0.4, high_hz=4.0, order=4)
baseline_processed = 0.5 * ppg1_processed + 0.5 * ppg2_processed

fig, axes = plt.subplots(4, 1, figsize=(11.8, 8.8), sharex=True, constrained_layout=True)
axes[0].plot(t, raw_ppg, color="#4c78a8", linewidth=1.0)
axes[0].set_title(f"Raw PPG mean signal: {DEMO_RECORD}, window {DEMO_WINDOW_INDEX}")
axes[0].set_ylabel("Raw amp.")

axes[1].plot(t, baseline_processed, color="#ff7f0e", linewidth=1.0)
axes[1].set_title("Baseline-processed PPG: 0.4-4 Hz band-pass + z-score")
axes[1].set_ylabel("Norm. amp.")

axes[2].plot(t, cleaned[start:stop], color="#d62728", linewidth=1.0)
axes[2].set_title("ACC-NLMS cleaned PPG")
axes[2].set_ylabel("Norm. amp.")

axes[3].plot(t, artifact[start:stop], color="#7f7f7f", linewidth=1.0)
axes[3].set_title("NLMS-estimated ACC-correlated artifact")
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("Artifact")

for ax in axes:
    ax.grid(True, alpha=0.25)

signal_path = PLOTS_DIR / f"{DEMO_RECORD}_raw_baseline_processed_acc_nlms_cleaned.png"
fig.savefig(signal_path, dpi=220)
plt.close(fig)

display(Image(filename=str(signal_path), width=920))
print("Saved:", signal_path)
""",
        "signal-code",
    ),
    md(
        """
## 9. 合并四种方法的窗口级结果
""",
        "merge-md",
    ),
    code(
        """
all_windows = comparison_windows.merge(
    temko_windows[["record", "center_time_s", "temko_wfpv_hr_bpm"]],
    on=["record", "center_time_s"],
    how="left",
).merge(
    temko_vd_windows[["record", "center_time_s", "temko_wfpv_vd_hr_bpm"]],
    on=["record", "center_time_s"],
    how="left",
)
all_windows.to_csv(OUTDIR / "four_methods_window_level_results.csv", index=False)
print("Saved:", OUTDIR / "four_methods_window_level_results.csv")
display(all_windows.head())
""",
        "merge-code",
    ),
    md(
        """
## 10. 与 BPM0 的时间序列对比图

该图展示 `BPM0`、PPG FFT、ACC-NLMS FFT、Temko WFPV online 和 Temko WFPV-VD offline 在同一记录中的时间序列。该图用于方法对比；论文正文中展示单一方法效果时，建议使用后面的 Temko-only 图。
""",
        "timeseries-md",
    ),
    code(
        """
PLOT_RECORD = "DATA_10_TYPE02"
record_df = all_windows[all_windows["record"] == PLOT_RECORD].copy()

fig, ax = plt.subplots(figsize=(12.5, 4.9), constrained_layout=True)
ax.plot(record_df["center_time_s"] / 60.0, record_df["hr_true_bpm"], label="BPM0 true HR", color="#1f77b4", linewidth=1.8)
ax.plot(record_df["center_time_s"] / 60.0, record_df["ppg_fft_hr_est_bpm"], label="PPG FFT", color="#ff7f0e", linewidth=1.0)
ax.plot(record_df["center_time_s"] / 60.0, record_df["ppg_acc_nlms_fft_hr_est_bpm"], label="ACC-NLMS FFT", color="#d62728", linewidth=1.15)
ax.plot(record_df["center_time_s"] / 60.0, record_df["temko_wfpv_hr_bpm"], label="Temko WFPV", color="#8c564b", linewidth=1.25)
ax.plot(record_df["center_time_s"] / 60.0, record_df["temko_wfpv_vd_hr_bpm"], label="Temko WFPV-VD offline", color="#9467bd", linewidth=1.35)
ax.set_title(f"Four HR estimation methods vs BPM0: {PLOT_RECORD}")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heart rate (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, ncol=2)

timeseries_path = PLOTS_DIR / f"{PLOT_RECORD}_hr_true_four_methods.png"
fig.savefig(timeseries_path, dpi=220)
plt.close(fig)

display(Image(filename=str(timeseries_path), width=980))
print("Saved:", timeseries_path)
""",
        "timeseries-code",
    ),
    md(
        """
## 11. Temko WFPV 单独与 BPM0 对比图

上一个图适合展示四种方法的相对差异，但曲线较多。论文中如果想突出 Temko WFPV 的效果，可以使用下面两张更干净的图：只保留 `BPM0` ground truth 和 Temko 估计心率。
""",
        "temko-only-timeseries-md",
    ),
    code(
        """
temko_only_df = record_df[["center_time_s", "hr_true_bpm", "temko_wfpv_hr_bpm"]].dropna().copy()
temko_only_error = temko_only_df["temko_wfpv_hr_bpm"] - temko_only_df["hr_true_bpm"]
temko_only_mae = np.mean(np.abs(temko_only_error))
temko_only_sdAE = np.std(np.abs(temko_only_error), ddof=1)
temko_only_r = np.corrcoef(temko_only_df["hr_true_bpm"], temko_only_df["temko_wfpv_hr_bpm"])[0, 1]

fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
ax.plot(
    temko_only_df["center_time_s"] / 60.0,
    temko_only_df["hr_true_bpm"],
    label="BPM0 true HR",
    color="#1f77b4",
    linewidth=2.1,
)
ax.plot(
    temko_only_df["center_time_s"] / 60.0,
    temko_only_df["temko_wfpv_hr_bpm"],
    label="Temko WFPV HR",
    color="#8c564b",
    linewidth=1.7,
)
ax.set_title(f"Temko WFPV HR estimation vs BPM0: {PLOT_RECORD}")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heart rate (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="upper left")
ax.text(
    0.985,
    0.06,
    f"MAE={temko_only_mae:.2f} bpm\\nsdAE={temko_only_sdAE:.2f} bpm\\nr={temko_only_r:.3f}",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 4},
)

temko_only_path = PLOTS_DIR / f"{PLOT_RECORD}_temko_wfpv_vs_bpm0_only.png"
fig.savefig(temko_only_path, dpi=220)
plt.close(fig)

display(Image(filename=str(temko_only_path), width=940))
print("Saved:", temko_only_path)
print(f"{PLOT_RECORD} Temko-only: MAE={temko_only_mae:.3f} bpm, sdAE={temko_only_sdAE:.3f} bpm, r={temko_only_r:.4f}")
""",
        "temko-only-timeseries-code",
    ),
    code(
        """
temko_vd_only_df = record_df[["center_time_s", "hr_true_bpm", "temko_wfpv_vd_hr_bpm"]].dropna().copy()
temko_vd_only_error = temko_vd_only_df["temko_wfpv_vd_hr_bpm"] - temko_vd_only_df["hr_true_bpm"]
temko_vd_only_mae = np.mean(np.abs(temko_vd_only_error))
temko_vd_only_sdAE = np.std(np.abs(temko_vd_only_error), ddof=1)
temko_vd_only_r = np.corrcoef(temko_vd_only_df["hr_true_bpm"], temko_vd_only_df["temko_wfpv_vd_hr_bpm"])[0, 1]

fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
ax.plot(
    temko_vd_only_df["center_time_s"] / 60.0,
    temko_vd_only_df["hr_true_bpm"],
    label="BPM0 true HR",
    color="#1f77b4",
    linewidth=2.1,
)
ax.plot(
    temko_vd_only_df["center_time_s"] / 60.0,
    temko_vd_only_df["temko_wfpv_vd_hr_bpm"],
    label="Temko WFPV-VD offline HR",
    color="#9467bd",
    linewidth=1.7,
)
ax.set_title(f"Temko WFPV-VD offline HR estimation vs BPM0: {PLOT_RECORD}")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Heart rate (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="upper left")
ax.text(
    0.985,
    0.06,
    f"MAE={temko_vd_only_mae:.2f} bpm\\nsdAE={temko_vd_only_sdAE:.2f} bpm\\nr={temko_vd_only_r:.3f}",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 4},
)

temko_vd_only_path = PLOTS_DIR / f"{PLOT_RECORD}_temko_wfpv_vd_vs_bpm0_only.png"
fig.savefig(temko_vd_only_path, dpi=220)
plt.close(fig)

display(Image(filename=str(temko_vd_only_path), width=940))
print("Saved:", temko_vd_only_path)
print(f"{PLOT_RECORD} Temko WFPV-VD offline: MAE={temko_vd_only_mae:.3f} bpm, sdAE={temko_vd_only_sdAE:.3f} bpm, r={temko_vd_only_r:.4f}")
""",
        "temko-vd-only-timeseries-code",
    ),
    md(
        """
## 12. 四种方法 MAE 对比图
""",
        "mae-md",
    ),
    code(
        """
records_for_plot = ["ALL_training", "ALL_competition", "ALL_DATABASE"]
labels_for_plot = ["Training", "Competition", "All"]
colors = {
    "PPG FFT": "#0072B2",
    "ACC-NLMS FFT": "#D55E00",
    "Temko WFPV": "#009E73",
    "Temko WFPV-VD offline": "#CC79A7",
}
x = np.arange(len(records_for_plot), dtype=float)
width = 0.18

fig, ax = plt.subplots(figsize=(10.4, 4.9), constrained_layout=True)
for i, method in enumerate(method_order):
    values = []
    for record_name in records_for_plot:
        row = metric_table[(metric_table["record"] == record_name) & (metric_table["method"] == method)]
        values.append(float(row["avAE_bpm"].iloc[0]) if not row.empty else np.nan)
    offset = (i - (len(method_order) - 1) / 2.0) * width
    ax.bar(x + offset, values, width=width, color=colors[method], label=method)
ax.set_xticks(x)
ax.set_xticklabels(labels_for_plot)
ax.set_ylabel("avAE / MAE (bpm)")
ax.set_title("DATABASE avAE comparison across four methods")
ax.grid(True, axis="y", alpha=0.25)
ax.legend(frameon=False)

mae_path = PLOTS_DIR / "mae_four_methods.png"
fig.savefig(mae_path, dpi=220)
plt.close(fig)

display(Image(filename=str(mae_path), width=850))
print("Saved:", mae_path)
""",
        "mae-code",
    ),
    md(
        """
## 13. Correlation 图

该图同时展示四种方法与 `BPM0` 的相关性。黑线表示理想估计。
""",
        "corr-md",
    ),
    code(
        """
specs = [
    ("PPG FFT", "ppg_fft_hr_est_bpm", "#0072B2"),
    ("ACC-NLMS FFT", "ppg_acc_nlms_fft_hr_est_bpm", "#D55E00"),
    ("Temko WFPV", "temko_wfpv_hr_bpm", "#009E73"),
    ("Temko WFPV-VD offline", "temko_wfpv_vd_hr_bpm", "#CC79A7"),
]

fig, ax = plt.subplots(figsize=(6.8, 6.1), constrained_layout=True)
vals = [all_windows["hr_true_bpm"].to_numpy()]
text_lines = []
for method, col, color in specs:
    valid = all_windows[["hr_true_bpm", col]].dropna()
    ax.scatter(valid["hr_true_bpm"], valid[col], s=13, alpha=0.36, label=method, color=color)
    vals.append(valid[col].to_numpy())
    row = metric_table[(metric_table["record"] == "ALL_DATABASE") & (metric_table["method"] == method)].iloc[0]
    text_lines.append(f"{method}: r={row['pearson_r']:.3f}, avAE={row['avAE_bpm']:.2f}")
finite = np.concatenate([v[np.isfinite(v)] for v in vals if v.size])
lo = float(finite.min() - 5)
hi = float(finite.max() + 5)
ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.text(0.04, 0.96, "\\n".join(text_lines), transform=ax.transAxes, va="top", fontsize=9)
ax.set_title("Correlation: estimated HR vs BPM0")
ax.set_xlabel("BPM0 true HR (bpm)")
ax.set_ylabel("Estimated HR (bpm)")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, loc="lower right")

corr_path = PLOTS_DIR / "correlation_four_methods.png"
fig.savefig(corr_path, dpi=220)
plt.close(fig)

display(Image(filename=str(corr_path), width=760))
print("Saved:", corr_path)
""",
        "corr-code",
    ),
    md(
        """
## 14. Bland-Altman 图

该图比较四种方法相对于 `BPM0` 的一致性和误差分布。
""",
        "ba-md",
    ),
    code(
        """
fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), constrained_layout=True)
for ax, (method, col, color) in zip(axes.ravel(), specs):
    valid = all_windows[["hr_true_bpm", col]].dropna()
    mean_hr = valid.mean(axis=1)
    diff_hr = valid[col] - valid["hr_true_bpm"]
    ax.scatter(mean_hr, diff_hr, s=10, alpha=0.30, color=color)
    row = all_metrics[(all_metrics["record"] == "ALL_DATABASE") & (all_metrics["method"] == method)].iloc[0]
    bias = float(row["bias_bpm"])
    lower = float(row["bland_altman_lower_bpm"])
    upper = float(row["bland_altman_upper_bpm"])
    ax.axhline(bias, color=color, linewidth=1.2, label="Bias")
    ax.axhline(lower, color="black", linestyle="--", linewidth=1.0, label="95% limits")
    ax.axhline(upper, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(method)
    ax.set_xlabel("Mean HR (bpm)")
    ax.set_ylabel("HR_est - BPM0 (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

ba_path = PLOTS_DIR / "bland_altman_four_methods.png"
fig.savefig(ba_path, dpi=220)
plt.close(fig)

display(Image(filename=str(ba_path), width=1000))
print("Saved:", ba_path)
""",
        "ba-code",
    ),
    md(
        """
## 15. 输出文件
""",
        "outputs-md",
    ),
    code(
        """
print("Four-method metric table:", OUTDIR / "four_methods_avAE_avRE_sdAE_table.csv")
print("Ultimate comparison table:", OUTDIR / "ultimate_four_method_comparison_table.csv")
print("Four-method window-level results:", OUTDIR / "four_methods_window_level_results.csv")
print("PPG/ACC-NLMS metrics:", OUTDIR / "ppg_nlms_acc_fft_metrics.csv")
print("Temko WFPV metrics:", OUTDIR / "temko_wfpv_database_metrics.csv")
print("Temko WFPV-VD offline metrics:", OUTDIR / "temko_wfpv_vd_database_metrics.csv")
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
