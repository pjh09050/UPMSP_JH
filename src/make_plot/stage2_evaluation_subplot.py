import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.Parameters import *
from src.common.pathConfig import *


def plot_single_evaluation_score(
    csv_dir,
    csv_filename,
    y_col="evaluation_score",
    label="Data",
    color="black",
    marker=".",
    linestyle="--",
):
    file_path = os.path.join(csv_dir, csv_filename)
    df = pd.read_csv(file_path)

    x = np.arange(len(df))
    y = pd.to_numeric(df[y_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(20, 12))

    ax.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        color=color,
        linewidth=2.5,
        markersize=10,
        label=label
    )

    ax.set_xlabel("Index", fontsize=36)
    ax.set_ylabel("Cumulative Reward", fontsize=36)
    ax.tick_params(axis='both', labelsize=28)
    # ax.legend(fontsize=28)
    plt.tight_layout()

    result_name = f"{label}_Stage2_Evaluation.png"
    save_path = os.path.join(csv_dir, result_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"{label} Evaluation score 그래프 저장 완료: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    Parameters.set_absolute_path()
    running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"

    if Parameters.save_plot:
        os.makedirs(running_data_save_directory, exist_ok=True)

    plot_single_evaluation_score(
        csv_dir=running_data_save_directory,
        csv_filename="251231_070716_makespan_two_stage_dqn_rule_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_agent2_evaluation_scores_makespan.csv",
        y_col="evaluation_score",
        label="Ours",
        color="black",
        marker=".",
        linestyle="--",
    )

    plot_single_evaluation_score(
        csv_dir=running_data_save_directory,
        csv_filename="260426_134010_makespan_two_stage_rnn_['M25_J1000_F7_P10-50_D5-10_Fgap_U1']_F7_rnn_second_stage_data.csv",
        y_col="test_cum_reward",
        label="RNN-based PPO",
        color="black",
        marker="^",
        linestyle="--",
    )