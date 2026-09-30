import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common.Parameters import *
from src.common.pathConfig import *
from matplotlib.ticker import MultipleLocator

def _draw_wave_break(ax, where="bottom", amp=0.015, waves=6, lw=1.8):
    x = np.linspace(0.02, 0.98, 300)
    y0 = 0.0 if where == "bottom" else 1.0
    y = y0 + amp * np.sin(2 * np.pi * waves * (x - x.min()) / (x.max() - x.min()))
    ax.plot(x, y, transform=ax.transAxes, clip_on=False, color="black", lw=lw)

def plot_from_csv_broken_y(
    csv_dir,
    csv_filename,
    y_col,
    ylabel,
    ylim_low,
    ylim_high,
    split_ratio=(3.0, 1.6),
    linestyle="--",
    marker=".",
    y_tick_step=500,
    show_high_y_ticks=True,
    force_high_ticks=True,      # ★ 추가: 상단 tick을 강제로 지정
    left_margin=0.12,           # ★ 추가: 왼쪽 여백(라벨/틱 잘림 방지)
):
    csv_path = os.path.join(csv_dir, csv_filename)
    df = pd.read_csv(csv_path)

    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' 이 CSV에 없습니다. 현재 컬럼: {list(df.columns)}")

    x = df["episode"].to_numpy() if "episode" in df.columns else np.arange(len(df))
    y = df[y_col].to_numpy(dtype=float)

    fig, (ax_high, ax_low) = plt.subplots(
        2, 1, sharex=True, figsize=(20, 12),
        gridspec_kw={"height_ratios": split_ratio}
    )

    ax_high.plot(x, y, linestyle=linestyle, marker=marker, color="black")
    ax_low.plot(x, y, linestyle=linestyle, marker=marker, color="black")

    ax_low.set_ylim(*ylim_low)
    ax_high.set_ylim(*ylim_high)

    # x 라벨
    ax_low.set_xlabel("Episodes", fontsize=28)

    # ★ 추가: x축을 0부터 시작 + xtick step 고정
    ax_low.set_xlim(left=0)
    ax_low.xaxis.set_major_locator(MultipleLocator(250))

    # ★ y 라벨은 하나만: fig.text로 고정(레이아웃에 덜 휘둘림)
    fig.text(left_margin * 0.45, 0.5, ylabel, va="center", rotation="vertical", fontsize=28)

    # tick 폰트
    # 상단 축: x tick mark + label 모두 제거 (끊김 구간에 보이는 세로선 제거)
    ax_high.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labelsize=24)
    # 하단 축: x tick은 아래만 유지(위쪽 tick mark 제거)
    ax_low.tick_params(axis="x", top=False, labelsize=24)
    ax_low.tick_params(axis="y", top=False, labelsize=24)
    ax_high.tick_params(axis="y", bottom=False, labelsize=24)

    # ★ 위/아래 y tick 간격 동일하게 강제
    if y_tick_step is not None:
        locator = MultipleLocator(y_tick_step)
        ax_high.yaxis.set_major_locator(locator)
        ax_low.yaxis.set_major_locator(locator)

    # ★ 상단 y tick 숫자 표시 여부
    ax_high.tick_params(axis="y", labelleft=show_high_y_ticks)

    # ★ 상단(6000~7800) tick을 "반드시" 찍히게 강제
    if force_high_ticks and y_tick_step is not None:
        start = int(np.ceil(ylim_high[0] / y_tick_step) * y_tick_step)
        end = int(np.floor(ylim_high[1] / y_tick_step) * y_tick_step)
        if start <= end:
            ax_high.set_yticks(np.arange(start, end + y_tick_step, y_tick_step))

    # 스파인 숨김
    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)

    # 물결 표시
    _draw_wave_break(ax_high, where="bottom", amp=0.018, waves=7, lw=2.0)
    _draw_wave_break(ax_low, where="top", amp=0.018, waves=7, lw=2.0)

    # 간격/레이아웃: left를 확실히 열어 tick 잘림 방지
    fig.subplots_adjust(hspace=0.3, left=left_margin, right=0.98, bottom=0.10, top=0.98)

    # 저장
    base = os.path.splitext(csv_filename)[0]
    save_name = f"{base}_{y_col}_brokenY.png"
    save_path = os.path.join(csv_dir, save_name)

    # ★ bbox_inches="tight"가 오히려 라벨을 자르는 케이스가 있어 pad_inches를 주고,
    # 필요하면 tight를 빼도 됨(아래 설정은 안전 쪽)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"그래프 저장 완료: {save_path}")

Parameters.set_absolute_path()
running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
if Parameters.save_plot:
    os.makedirs(running_data_save_directory, exist_ok=True)

plot_from_csv_broken_y(
    csv_dir=running_data_save_directory,
    csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_family_running_data.csv",
    y_col="last_train_50",
    ylabel="Average Makespan in last 50 episodes",
    ylim_low=(2000, 2800),
    ylim_high=(5000, 7800),
    y_tick_step=500,
    show_high_y_ticks=True
)
