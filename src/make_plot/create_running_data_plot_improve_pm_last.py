import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common.Parameters import *
from src.common.pathConfig import *
from matplotlib.ticker import MultipleLocator

def plot_improvement_percent_from_csv_pm_last(
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

    for df_name, df in [("baseline(pm)", df_base), ("ours(family)", df_ours)]:
        if y_col not in df.columns:
            raise ValueError(f"[{df_name}] y_col '{y_col}' 이 CSV에 없습니다. 현재 컬럼: {list(df.columns)}")

    # --- get pm last value (based on last episode if exists, else last row)
    if "episode" in df_base.columns:
        df_base_sorted = df_base.sort_values("episode")
        pm_last = float(df_base_sorted[y_col].iloc[-1])
        pm_last_ep = int(df_base_sorted["episode"].iloc[-1])
    else:
        pm_last = float(df_base[y_col].iloc[-1])
        pm_last_ep = None

    # --- x and ours series (family)
    if "episode" in df_ours.columns:
        df_ours_sorted = df_ours.sort_values("episode")
        x = df_ours_sorted["episode"].to_numpy()
        ours = df_ours_sorted[y_col].to_numpy(dtype=float)
    else:
        ours = df_ours[y_col].to_numpy(dtype=float)
        x = np.arange(len(ours))

    # --- improvement vs pm_last (%)
    eps = 1e-12
    improvement_pct = (pm_last - ours) / (pm_last + eps) * 100.0

    # --- plot
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(x, improvement_pct, linestyle="-", linewidth=5, color="black")

    ax.set_xlabel("The cumulative number of episodes", fontsize=36)
    ax.set_xlim(left=0, right=2000)
    # y-label 문구는 "pm last 대비"가 명확하게
    ax.set_ylabel("Makespan improvement rate over PMC (%)", fontsize=36)

    ax.tick_params(axis="both", labelsize=28)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))
    if x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))

    ax.axhline(0, color="gray", linestyle=":", linewidth=5)
    plt.tight_layout()

    # save
    save_name = "improvement_last_episode.png"
    save_path = os.path.join(csv_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"그래프 저장 완료: {save_path}")
    print(f"[INFO] PM last ({'episode '+str(pm_last_ep) if pm_last_ep is not None else 'last row'}): {pm_last:.6f}")
    print(f"[INFO] Last improvement: {improvement_pct[-1]:.3f}%")
    print(f"[INFO] Best improvement: {np.nanmax(improvement_pct):.3f}%")
    print(f"[INFO] Worst improvement: {np.nanmin(improvement_pct):.3f}%")

Parameters.set_absolute_path()
running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"

plot_improvement_percent_from_csv_pm_last(
    csv_dir=running_data_save_directory,
    baseline_csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_pm_running_data.csv",
    ours_csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_family_running_data.csv",
    y_col="last_train_50",
    ylim=None,
    y_tick_step=10,
    x_tick_step=250
)
