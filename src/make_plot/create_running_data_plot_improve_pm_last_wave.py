import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.common.Parameters import *
from src.common.pathConfig import *

def _draw_wave_break(ax, where="bottom", amp=0.015, waves=6, lw=1.8):
    """Draw a wavy break marker across the axis in Axes coordinates."""
    x = np.linspace(0.02, 0.98, 300)
    y0 = 0.0 if where == "bottom" else 1.0
    y = y0 + amp * np.sin(2 * np.pi * waves * (x - x.min()) / (x.max() - x.min()))
    ax.plot(x, y, transform=ax.transAxes, clip_on=False, color="black", lw=lw)

def plot_improvement_vs_pmc_last_brokenY(
    csv_dir,
    baseline_csv_filename,   # PMC (pm)
    ours_csv_filename,       # family (JA)
    y_col,
    ylim_low=(-150, -60),
    ylim_high=(-10, 20),
    split_ratio=(1.0, 1.0),
    y_tick_step_low=20,
    y_tick_step_high=10,
    x_tick_step=250,
    line_width=4,
    left_margin=0.15,
    show_high_y_ticks=True,
    waves=7,
    amp=0.018,
):
    # --- load
    base_path = os.path.join(csv_dir, baseline_csv_filename)
    ours_path = os.path.join(csv_dir, ours_csv_filename)

    df_base = pd.read_csv(base_path)
    df_ours = pd.read_csv(ours_path)

    for df_name, df in [("baseline(pm)", df_base), ("ours(family)", df_ours)]:
        if y_col not in df.columns:
            raise ValueError(f"[{df_name}] y_col '{y_col}' 이 CSV에 없습니다. 현재 컬럼: {list(df.columns)}")

    # --- pm_last (PMC final reference)
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

    # --- plot (broken y-axis with wave)
    fig, (ax_high, ax_low) = plt.subplots(
        2, 1, sharex=True, figsize=(20, 12),
        gridspec_kw={"height_ratios": split_ratio}
    )

    ax_high.plot(x, improvement_pct, linestyle="-", linewidth=line_width, color="black")
    ax_low.plot(x, improvement_pct, linestyle="-", linewidth=line_width, color="black")

    # y-lims
    ax_low.set_ylim(*ylim_low)
    ax_high.set_ylim(*ylim_high)

    # x-axis
    ax_low.set_xlabel("The cumulative number of episodes", fontsize=36)
    ax_low.set_xlim(left=0, right=2000)
    if x_tick_step is not None:
        ax_low.xaxis.set_major_locator(MultipleLocator(x_tick_step))

    # one shared y-label (stable)
    fig.text(left_margin * 0.45, 0.5, "Makespan improvement rate over PMC (%)", va="center", rotation="vertical", fontsize=36)

    # ticks
    ax_high.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labelsize=28)
    ax_low.tick_params(axis="x", top=False, labelsize=28)
    ax_low.tick_params(axis="y", top=False, labelsize=28)
    ax_high.tick_params(axis="y", bottom=False, labelsize=28)

    ax_high.tick_params(axis="y", labelleft=show_high_y_ticks)

    # y tick steps
    if y_tick_step_high is not None:
        ax_high.yaxis.set_major_locator(MultipleLocator(y_tick_step_high))
    if y_tick_step_low is not None:
        ax_low.yaxis.set_major_locator(MultipleLocator(y_tick_step_low))

    # reference line (0%) on high axis only
    ax_high.axhline(0, color="gray", linestyle=":", linewidth=4)

    # spines + wave breaks
    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)

    _draw_wave_break(ax_high, where="bottom", amp=amp, waves=waves, lw=2.0)
    _draw_wave_break(ax_low, where="top", amp=amp, waves=waves, lw=2.0)

    # layout
    fig.subplots_adjust(hspace=0.2, left=left_margin, right=0.98, bottom=0.10, top=0.98)

    # save
    save_name = "improvement_last_episode_brokenY_wave.png"
    save_path = os.path.join(csv_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"그래프 저장 완료: {save_path}")
    print(f"[INFO] PMC final ({'episode '+str(pm_last_ep) if pm_last_ep is not None else 'last row'}): {pm_last:.6f}")
    print(f"[INFO] Final improvement: {improvement_pct[-1]:.3f}%")
    print(f"[INFO] Best improvement: {np.nanmax(improvement_pct):.3f}%")
    print(f"[INFO] Worst improvement: {np.nanmin(improvement_pct):.3f}%")

    return save_path


# --------- run ----------
Parameters.set_absolute_path()
running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
if Parameters.save_plot:
    os.makedirs(running_data_save_directory, exist_ok=True)

plot_improvement_vs_pmc_last_brokenY(
    csv_dir=running_data_save_directory,
    baseline_csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_pm_running_data.csv",
    ours_csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_family_running_data.csv",
    y_col="last_train_50",
    ylim_low=(-150, -60),
    ylim_high=(-10, 20),
    y_tick_step_low=20,
    y_tick_step_high=10,
    x_tick_step=250,
    line_width=4,
    show_high_y_ticks=True,
    waves=7,
    amp=0.018,
)
