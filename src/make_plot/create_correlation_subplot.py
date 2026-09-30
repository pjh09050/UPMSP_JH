import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.Parameters import *
from src.common.pathConfig import *


def plot_single_train_score_correlation(
    csv_dir,
    csv_filename,
    train_col="train_data",
    score_col="score_list",
    label="Data",
    color="black",
    marker="o",
):
    file_path = os.path.join(csv_dir, csv_filename)
    df = pd.read_csv(file_path)

    train_data = pd.to_numeric(df[train_col], errors="coerce")
    score_list = pd.to_numeric(df[score_col], errors="coerce")

    valid_corr = (~train_data.isna()) & (~score_list.isna())

    if valid_corr.sum() < 2:
        print(f"{label}: 상관계수를 계산할 유효한 데이터가 부족합니다.")
        return

    train_data_valid = train_data[valid_corr]
    score_list_valid = score_list[valid_corr]

    correlation = np.corrcoef(train_data_valid, score_list_valid)[0, 1]

    fig, ax = plt.subplots(figsize=(20, 12))

    ax.scatter(
        train_data_valid,
        score_list_valid,
        color=color,
        marker=marker,
        s=80,
        alpha=0.7,
        label=f"Correlation: {correlation:.2f})"
    )

    ax.set_xlabel("Makespan", fontsize=36)
    ax.set_ylabel("Cumulative Reward", fontsize=36)
    ax.tick_params(axis="both", labelsize=28)
    ax.legend(fontsize=28)
    plt.tight_layout()

    result_name = f"{label}_Stage2_Cor.png"
    save_path = os.path.join(csv_dir, result_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{label} Correlation Scatter 그래프 저장 완료: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    Parameters.set_absolute_path()
    running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"

    if Parameters.save_plot:
        os.makedirs(running_data_save_directory, exist_ok=True)

    plot_single_train_score_correlation(
        csv_dir=running_data_save_directory,
        csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_family_running_data.csv",
        train_col="train_data",
        score_col="score_list",
        label="Ours",
        color="black",
        marker="o",
    )

    plot_single_train_score_correlation(
        csv_dir=running_data_save_directory,
        csv_filename="260426_134010_makespan_two_stage_rnn_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_F7_rnn_second_stage_data.csv",
        train_col="train_makespan",
        score_col="train_cum_reward",
        label="RNN-based PPO",
        color="black",
        marker="^",
    )