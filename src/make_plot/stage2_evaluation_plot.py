import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.Parameters import *
from src.common.pathConfig import *


def plot_evaluation_score_two_csv(
    csv_dir,
    csv_filename_1,
    csv_filename_2,
    x_col_1="episode",
    y_col_1="evaluation_score",
    x_col_2="episode",
    y_col_2="evaluation_score",
    label_1="Data 1",
    label_2="Data 2",
    color_1="black",
    color_2="red",
    marker_1=".",
    marker_2="^",
    linestyle_1="--",
    linestyle_2="--",
):
    file_path_1 = os.path.join(csv_dir, csv_filename_1)
    file_path_2 = os.path.join(csv_dir, csv_filename_2)

    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    x1 = np.arange(len(df1))
    y1 = pd.to_numeric(df1[y_col_1], errors="coerce")

    x2 = np.arange(len(df2))
    y2 = pd.to_numeric(df2[y_col_2], errors="coerce")

    fig, ax = plt.subplots(figsize=(20, 12))

    ax.plot(
        x1,
        y1,
        marker=marker_1,
        linestyle=linestyle_1,
        color=color_1,
        linewidth=2.5,
        markersize=10,
        label=label_1
    )

    ax.plot(
        x2,
        y2,
        marker=marker_2,
        linestyle=linestyle_2,
        color=color_2,
        linewidth=2.5,
        markersize=10,
        label=label_2
    )

    ax.set_xlabel("Episode / Index", fontsize=36)
    ax.set_ylabel("Cumulative Reward", fontsize=36)
    ax.tick_params(axis='both', labelsize=28)
    # ax.legend(fontsize=28)
    plt.tight_layout()

    result_name = f"compare_{label_1}_{label_2}_Stage2_Evaluation.png"
    save_path = os.path.join(csv_dir, result_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Evaluation score 그래프 저장 완료: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    Parameters.set_absolute_path()
    running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"

    if Parameters.save_plot:
        os.makedirs(running_data_save_directory, exist_ok=True)

    plot_evaluation_score_two_csv(
        csv_dir=running_data_save_directory,
        csv_filename_1="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_agent2_evaluation_scores_makespan.csv",
        csv_filename_2="260430_224805_makespan_two_stage_rnn_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_F7_rnn_second_stage_data.csv",
        x_col_1="episode",
        y_col_1="evaluation_score",
        x_col_2="episode",
        y_col_2="test_cum_reward",
        label_1="Makespan",
        label_2="Reward",
        color_1="black",
        color_2="black",
        marker_1=".",
        marker_2="^",
    )