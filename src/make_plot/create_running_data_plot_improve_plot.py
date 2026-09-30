import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.common.Parameters import *
from src.common.pathConfig import *

def plot_improvement_percent_from_csv(
    csv_dir,
    baseline_csv_filename,   # pm (A)
    ours_csv_filename,       # family (B)
    y_col,
    ylim=None,
    y_tick_step=None,
    x_tick_step=None,
):
    # --- load
    base_path = os.path.join(csv_dir, baseline_csv_filename)
    ours_path = os.path.join(csv_dir, ours_csv_filename)

    df_base = pd.read_csv(base_path)
    df_ours = pd.read_csv(ours_path)

    for df_name, df in [("baseline", df_base), ("ours", df_ours)]:
        if y_col not in df.columns:
            raise ValueError(f"[{df_name}] y_col '{y_col}' 이 CSV에 없습니다. 현재 컬럼: {list(df.columns)}")

    # --- align by episode if exists, else by index
    if "episode" in df_base.columns and "episode" in df_ours.columns:
        df_base2 = df_base[["episode", y_col]].rename(columns={y_col: "baseline"})
        df_ours2 = df_ours[["episode", y_col]].rename(columns={y_col: "ours"})
        dfm = pd.merge(df_base2, df_ours2, on="episode", how="inner").sort_values("episode")
        x = dfm["episode"].to_numpy()
        baseline = dfm["baseline"].to_numpy(dtype=float)
        ours = dfm["ours"].to_numpy(dtype=float)
    else:
        n = min(len(df_base), len(df_ours))
        x = np.arange(n)
        baseline = df_base[y_col].to_numpy(dtype=float)[:n]
        ours = df_ours[y_col].to_numpy(dtype=float)[:n]

    # --- improvement (%)
    eps = 1e-12
    improvement_pct = (baseline - ours) / (baseline + eps) * 100.0

    # --- plot
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(x, improvement_pct, linestyle="-", linewidth=5, color="black") # marker="."

    ax.set_xlabel("Episodes", fontsize=28)
    ax.set_ylabel("Relative Improvement of JA over PMC (%)",fontsize=28)
    ax.tick_params(axis="both", labelsize=24)

    # y-limit
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # y tick step
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))

    # x-axis from 0
    # ax.set_xlim(left=0, right=2000)

    # x tick step
    if x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))

    ax.axhline(0, color="gray", linestyle=":", linewidth=5)
    plt.tight_layout()

    # save
    save_name = f"improvement_each_episode.png"
    save_path = os.path.join(csv_dir, save_name)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장 완료: {save_path}")

    # 간단 요약(마지막 값/최대 개선 등)도 같이 출력
    print(f"[INFO] Last improvement: {improvement_pct[-1]:.3f}%")
    print(f"[INFO] Best improvement: {np.nanmax(improvement_pct):.3f}%")
    print(f"[INFO] Worst improvement: {np.nanmin(improvement_pct):.3f}%")

Parameters.set_absolute_path()
running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
if Parameters.save_plot:
    os.makedirs(running_data_save_directory, exist_ok=True)

# baseline = pm (A), ours = family (B)
plot_improvement_percent_from_csv(
    csv_dir=running_data_save_directory,
    baseline_csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_pm_running_data.csv",
    ours_csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_family_running_data.csv",
    y_col="last_train_50",
    ylim=None,          # 예: (0, 10) 처럼 좁히면 더 강조됨
    y_tick_step=5,      # 예: 1% 단위
    x_tick_step=250
)
