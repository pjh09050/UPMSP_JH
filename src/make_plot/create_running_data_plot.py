import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src.common.Parameters import *
from src.common.pathConfig import *

def plot_from_csv(csv_dir, csv_filename, y_col, ylabel, ylim, y_tick_step=None, x_tick_step=None):
    csv_path = os.path.join(csv_dir, csv_filename)
    df = pd.read_csv(csv_path)

    if y_col not in df.columns:
        raise ValueError(
            f"y_col '{y_col}' 이 CSV에 없습니다. 현재 컬럼: {list(df.columns)}"
        )

    x = df["episode"].to_numpy() if "episode" in df.columns else np.arange(len(df))
    y = df[y_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(x, y, linestyle="--", marker=".", color="black")

    ax.set_xlabel("Episodes", fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="both", labelsize=24)

    # y-limit
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        y_min, y_max = ax.get_ylim()
        print(f"[INFO] Auto y-lim used: ({round(y_min, 2)}, {round(y_max, 2)})")

    # y tick step
    if y_tick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))

    # ★ x축을 0부터 시작하게
    ax.set_xlim(left=0)

    # ★ x tick step을 주면 0, step, 2*step ... 으로 찍힘
    if x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))

    plt.tight_layout()

    base = os.path.splitext(csv_filename)[0]
    save_name = f"{base}_{y_col}.png"
    save_path = os.path.join(csv_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"그래프 저장 완료: {save_path}")

Parameters.set_absolute_path()
running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
if Parameters.save_plot:
    os.makedirs(running_data_save_directory, exist_ok=True)

plot_from_csv(
    csv_dir=running_data_save_directory,    csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_pm_running_data.csv",
    y_col="last_train_50",
    ylabel="Average Makespan in last 50 episodes",
    ylim=(1850, 8000),
    y_tick_step=500,
    x_tick_step=250        # 예: 0, 250, 500, ...
)
