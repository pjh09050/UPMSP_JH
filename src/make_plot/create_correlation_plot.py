import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from src.common.Parameters import *
from src.common.pathConfig import *


def plot_train_score_correlation_two_csv(
    csv_dir,
    csv_filename_1,
    csv_filename_2,
    train_col_1="train_data",
    score_col_1="score_list",
    train_col_2="train_data",
    score_col_2="score_list",
    label_1="Data 1",
    label_2="Data 2",
    color_1="black",
    color_2="red",
):
    file_path_1 = os.path.join(csv_dir, csv_filename_1)
    file_path_2 = os.path.join(csv_dir, csv_filename_2)

    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    train_data_1 = pd.to_numeric(df1[train_col_1], errors="coerce")
    score_list_1 = pd.to_numeric(df1[score_col_1], errors="coerce")
    valid_corr_1 = (~train_data_1.isna()) & (~score_list_1.isna())

    train_data_2 = pd.to_numeric(df2[train_col_2], errors="coerce")
    score_list_2 = pd.to_numeric(df2[score_col_2], errors="coerce")
    valid_corr_2 = (~train_data_2.isna()) & (~score_list_2.isna())

    if valid_corr_1.sum() < 2:
        print(f"{label_1}: 상관계수를 계산할 유효한 데이터가 부족합니다.")
        return

    if valid_corr_2.sum() < 2:
        print(f"{label_2}: 상관계수를 계산할 유효한 데이터가 부족합니다.")
        return

    correlation_1 = np.corrcoef(train_data_1[valid_corr_1], score_list_1[valid_corr_1])[0, 1]
    correlation_2 = np.corrcoef(train_data_2[valid_corr_2], score_list_2[valid_corr_2])[0, 1]

    fig, ax = plt.subplots(figsize=(20, 12))

    ax.scatter(
        train_data_1[valid_corr_1],
        score_list_1[valid_corr_1],
        color=color_1,
        marker='o',
        s=80,
        alpha=0.7,
        label=f'{label_1} (Corr: {correlation_1:.2f})'
    )

    ax.scatter(
        train_data_2[valid_corr_2],
        score_list_2[valid_corr_2],
        color=color_2,
        marker='^',
        s=80,
        alpha=0.7,
        label=f'{label_2} (Corr: {correlation_2:.2f})'
    )

    ax.set_xlabel('Makespan', fontsize=36)
    ax.set_ylabel('Cumulative Reward', fontsize=36)
    ax.tick_params(axis='both', labelsize=28)
    ax.legend(fontsize=28)
    plt.tight_layout()

    result_name = f"compare_{label_1}_{label_2}_Stage2_Cor.png"
    save_path = os.path.join(csv_dir, result_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Correlation Scatter 그래프 저장 완료: {save_path}")
    plt.close(fig)


Parameters.set_absolute_path()
running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"

plot_train_score_correlation_two_csv(
    csv_dir=running_data_save_directory,
    csv_filename_1="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_A4_F7_family_running_data.csv",
    csv_filename_2="260430_224805_makespan_two_stage_rnn_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_F7_rnn_second_stage_data.csv",
    train_col_1="train_data",
    score_col_1="score_list",
    train_col_2="train_makespan",
    score_col_2="train_cum_reward",
    label_1="Ours",
    label_2="RNN-based PPO",
    color_1="black",
    color_2="red",
)