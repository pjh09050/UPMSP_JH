import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from src.simlator.Simulator import *
from src.common.Parameters import *
import re
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MonteCarloNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MonteCarloNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

class MC_Rule_Select:
    batch_size = 64

    @classmethod
    def train(cls, model, optimizer, memory, loss_list):
        batch = random.sample(memory, cls.batch_size)  # 배치 단위로 에피소드 샘플링
        for episode in batch:
            G = 0
            states, actions, targets = [], [], []

            for state, action, reward in reversed(episode):
                G = reward + Hyperparameters.gamma * G  # 누적 보상 계산
                states.append(state)
                actions.append(action)
                targets.append(G)

            states.reverse()
            actions.reverse()
            targets.reverse()

            state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            action_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
            target_tensor = torch.tensor(np.array(targets), dtype=torch.float32).to(device)

            # Q-values 계산 및 손실 계산
            q_values = model(state_tensor)
            q_values = q_values.gather(1, action_tensor).squeeze(1)
            loss = nn.HuberLoss()(q_values, target_tensor)

            # 최적화 수행
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @classmethod
    def get_action_list(cls, state, model):
        if isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0).to(device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = model(state_tensor).squeeze(0).cpu().numpy()
        return q_values

    @classmethod
    def main(cls):
        min_score_episode = cls.run_learning_phase()
        makespan = cls.run_result_phase(min_score_episode)
        return round(makespan, 3)

    @classmethod
    def run_learning_phase(cls):
        env = Simulator()
        model = MonteCarloNN(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.learning_rate)
        memory = deque(maxlen=cls.batch_size)

        makespan_list = []
        last_50_makespan = []
        score_list = []
        last_50_score = []
        tardiness_list = []
        last_50_tardiness = []
        loss_list = []

        test_start = True
        test_makespan_list = []
        last_50_test_makespan = []
        test_score_list = []
        last_50_test_score = []
        test_tardiness_list = []
        last_50_test_tardiness = []

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}_F{Hyperparameters.family_type}_dsp_rule"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)

        plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(plot_save_directory, exist_ok=True)

        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / (Hyperparameters.episode - 1))
            pm_count = 0
            score = 0.0
            for dataid in Parameters.datasetId:
                s = env.reset(dataid)
                done = False
                score = 0.0
                pm_count = 0
                episode = []
                while not done:
                    s_tensor = torch.tensor(s, dtype=torch.float32).to(device)
                    a_list = cls.get_action_list(s_tensor, model)
                    s_prime, a, r, done = env.rule_select(a_list, epsilon)
                    if a == len(a_list)-1:
                        pm_count += 1
                    episode.append((s, a, r))
                    s = s_prime
                    score += r
                    if done:
                        break

                memory.append(episode)
            if len(memory) >= cls.batch_size:
                cls.train(model, optimizer, memory, loss_list)
                makespan_list, tardiness_list, score_list = cls.script_performance(env, n_epi, epsilon, memory, score, True, makespan_list, tardiness_list, score_list, pm_count, 'train')
                last_50_makespan.append(np.mean(makespan_list[-50:]))
                last_50_score.append(np.mean(score_list[-50:]))
                last_50_tardiness.append(np.mean(tardiness_list[-50:]))
                if Parameters.param_down_on:
                    file_name = str(n_epi) + "param.pt"
                    file_path = os.path.join(save_directory, file_name)
                    torch.save(model.state_dict(), file_path)
            else:
                output_string = f"no train: n_episode: {n_epi}, n_buffer: {len(memory)}, eps: {epsilon * 100:.1f}%, pm count: {pm_count}"
                print(output_string)

            if test_start == True:
                test_score = 0.0
                for dataid in Parameters.datasetId:
                    epsilon = 0.0  # 테스트는 epsilon = 0으로 고정
                    s = env.reset(dataid)
                    done = False
                    test_score = 0.0
                    pm_count = 0
                    while not done:
                        s_tensor = torch.tensor(s, dtype=torch.float32).to(device)
                        a_list = cls.get_action_list(s_tensor, model)
                        s_prime, a, r, done = env.rule_select(a_list, epsilon)
                        if a == len(a_list) - 1:
                            pm_count += 1
                        s = s_prime
                        test_score += r
                        if done:
                            break

                test_makespan_list, test_tardiness_list, test_score_list = cls.script_performance(env, n_epi, epsilon, memory, test_score, True,test_makespan_list, test_tardiness_list, test_score_list, pm_count, 'test')
                last_50_test_makespan.append(np.mean(test_makespan_list[-50:]))
                last_50_test_score.append(np.mean(test_score_list[-50:]))
                last_50_test_tardiness.append(np.mean(test_tardiness_list[-50:]))

        # 학습 및 테스트 결과를 2열 3행 subplot에 그리기
        object = Hyperparameters.object
        print('object:', object)
        if object == 'total_tardiness':
            object = 'Total Tardiness'
            train_data = tardiness_list
            test_data = test_tardiness_list
            last_train_50 = last_50_tardiness
            last_test_50 = last_50_test_tardiness
        else:
            object = 'Makespan'
            train_data = makespan_list
            test_data = test_makespan_list
            last_train_50 = last_50_makespan
            last_test_50 = last_50_test_makespan

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(68, 42))
        # 학습 목적함수 (왼쪽 첫 번째)
        axs[0, 0].plot(range(len(train_data)), train_data, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 0].set_xlabel('Episodes', fontsize=36)
        axs[0, 0].set_ylabel(f'Training {object}', fontsize=36)
        axs[0, 0].tick_params(axis='both', labelsize=28)

        # 학습 Score (왼쪽 두 번째)
        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='blue', label='Training Score')
        axs[1, 0].set_xlabel('Episodes', fontsize=36)
        axs[1, 0].set_ylabel('Training Score', fontsize=36)
        axs[1, 0].tick_params(axis='both', labelsize=28)

        axs[0, 1].plot(range(len(last_train_50)), last_train_50, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 1].set_xlabel('Episodes', fontsize=36)
        axs[0, 1].set_ylabel(f'Average {object} over previous 50 episodes', fontsize=36)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        axs[1, 1].plot(range(len(last_50_score)), last_50_score, linestyle='--', marker='.', color='blue', label=f'Training Score')
        axs[1, 1].set_xlabel('Episodes', fontsize=36)
        axs[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=36)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        # 학습 Loss (왼쪽 세 번째)
        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red', label='Training Loss')
        axs[2, 0].set_xlabel('Episodes', fontsize=36)
        axs[2, 0].set_ylabel('Training Loss', fontsize=36)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        # 테스트 Score, 목적함수 상관관계 (오른쪽 세 번째)
        correlation = np.corrcoef(train_data, score_list)[0, 1]
        axs[2, 1].scatter(train_data, score_list, color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        axs[2, 1].set_xlabel(f'{object}', fontsize=36)
        axs[2, 1].set_ylabel('Score', fontsize=36)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        axs[2, 1].legend(fontsize=28)
        fig.tight_layout()

        if test_start == True:
            fig2, axs2 = plt.subplots(2, 2, figsize=(58, 26))
            fig2.suptitle('Test', fontsize=48)
            # 테스트 목적함수 (오른쪽 첫 번째)
            axs2[0, 0].plot(range(len(test_data)), test_data, linestyle='--', marker='.', color='purple', label=f'Test {object}')
            axs2[0, 0].set_xlabel('Test Episodes', fontsize=36)
            axs2[0, 0].set_ylabel(f'Test {object}', fontsize=36)
            axs2[0, 0].tick_params(axis='both', labelsize=28)
            # 테스트 Score (오른쪽 두 번째)
            axs2[1, 0].plot(range(len(test_score_list)), test_score_list, linestyle='-', marker='.', color='orange', label='Test Score')
            axs2[1, 0].set_xlabel('Test Episodes', fontsize=36)
            axs2[1, 0].set_ylabel('Test Score', fontsize=36)
            axs2[1, 0].tick_params(axis='both', labelsize=28)
            # 테스트 목적함수 (오른쪽 첫 번째)
            axs2[0, 1].plot(range(len(last_test_50)), last_test_50, linestyle='--', marker='.', color='purple', label=f'Test {object}')
            axs2[0, 1].set_xlabel('Test Episodes', fontsize=36)
            axs2[0, 1].set_ylabel(f'Test {object}', fontsize=36)
            axs2[0, 1].tick_params(axis='both', labelsize=28)
            # 테스트 Score (오른쪽 두 번째)
            axs2[1, 1].plot(range(len(last_50_test_score)), last_50_test_score, linestyle='-', marker='.', color='orange', label='Test Score')
            axs2[1, 1].set_xlabel('Episodes', fontsize=36)
            axs2[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=36)
            axs2[1, 1].tick_params(axis='both', labelsize=28)
            fig2.legend(fontsize=28)
            fig2.tight_layout()

        if Parameters.save_plot:
            result_name_1 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Parameters.datasetId}_F{Hyperparameters.family_type}_{Hyperparameters.action_type}_train_result.png"
            file_path_1 = os.path.join(plot_save_directory, result_name_1)
            fig.savefig(file_path_1)
            print(f"첫 번째 그래프가 저장되었습니다: {file_path_1}")
            if test_start == True:
                result_name_2 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Parameters.datasetId}_F{Hyperparameters.family_type}_{Hyperparameters.action_type}_test_result.png"
                file_path_2 = os.path.join(plot_save_directory, result_name_2)
                fig2.savefig(file_path_2)
                print(f"두 번째 그래프가 저장되었습니다: {file_path_2}")
        # plt.show()
        min_test_score = max(test_score_list)
        min_score_episode = test_score_list.index(min_test_score)
        print(f"테스트 중 가장 낮은 score는 {min_test_score:.2f}이며, 해당 episode는 {min_score_episode}입니다.")
        # env.gantt_chart(score_list[-1], 'dsp_rule')
        print("학습 및 테스트가 종료되었습니다.")
        return min_score_episode

    @classmethod
    def run_result_phase(cls, min_score_episode):
        checkpoint_path = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}_F{Hyperparameters.family_type}_dsp_rule"
        checkpoint_file = f"{min_score_episode}param.pt"
        checkpoint_full_path = os.path.join(checkpoint_path, checkpoint_file)
        makespan = cls.get_result(checkpoint_full_path, Parameters.datasetId)
        return round(makespan, 3)

    @classmethod
    def get_result(cls, parameter, dataSets):
        env = Simulator
        model = MonteCarloNN(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        params = torch.load(parameter)
        model.load_state_dict(params)
        model.eval()
        for data_id in dataSets:
            s = env.reset(data_id)
            done = False
            score = 0.0
            while not done:
                epsilon = 0
                s_tensor = torch.tensor(s, dtype=torch.float32).to(device)
                a_list = cls.get_action_list(s_tensor, model)
                s_prime, a, r, done = env.rule_select(a_list, epsilon)
                s = s_prime
                score += r
                if done:
                    break
            Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, t_max = env.performance_measure()
            env.gantt_chart(score, 'dsp_rule')

            print(f"dataset: {data_id}")
            print(f"score: {score}")
            print("util:", util)
            print("Total Flow time:", round(Total_Flow_time, 3))
            print("makespan:", round(makespan, 3))
            print("Total Lateness time:", round(Total_Lateness_time, 3))
            print("Total Tardiness time:", round(Total_Tardiness_time, 3))
            print("T_max:", round(t_max, 3))
            if Parameters.log_on:
                logging.info(f"dataset: {data_id}")
                logging.info(f"util: {util}")
            if Parameters.save_plot:
                plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
                env.write_reward_txt(makespan, score, plot_save_directory)
            print("평가가 종료되었습니다.")
            return round(makespan, 3)

    @classmethod
    def get_evaluate(cls, checkpoint_path, number_of_checkpoint, point_num, datasets):
        env = Simulator()
        file_list = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))]
        interver = len(file_list)//number_of_checkpoint
        check_point_list = [i for i in range(1, len(file_list)) if i%interver == 0 ]
        check_point_list.append(len(file_list)-1)

        model = MonteCarloNN(Hyperparameters.input_layer, Hyperparameters.output_layer)  # Initialize neural network
        mean_reward_by_checkpoint = {}
        max_reward_by_checkpoint = {}

        best_makespan = float('inf')
        best_checkpoint = None

        for check_point_number in check_point_list[-point_num:]:
            check_point = f"{checkpoint_path}/{check_point_number}param.pt"
            params = torch.load(check_point)
            model.load_state_dict(params)
            model.eval()
            reward_list = []
            for dataset in datasets:
                s = env.reset(dataset)
                done = False
                score = 0.0
                while not done:
                    epsilon = 0
                    a = cls.get_select_action(s, model)
                    s_prime, r, done = env.rule_select(a)
                    s = s_prime
                    score += r
                    if done:
                        break

                Flow_time, util, makespan, tardiness, lateness, t_max = env.performance_measure()

                if makespan <= best_makespan:
                    best_makespan = makespan
                    best_checkpoint = check_point_number

                reward_list.append(round(score, 3))
                if Parameters.log_on:
                    logging.info(f"checkpoint: {check_point_number}")
                    logging.info(f"dataset: {dataset}")
                    logging.info(f"score:{round(score, 3)}")

                print(f"checkpoint: {check_point_number}")
                print(f"dataset: {dataset}")
                print(f"score:{score}")

            mean_reward_by_checkpoint[check_point_number] = sum(reward_list) / len(reward_list)
            max_reward_by_checkpoint[check_point_number] = max(reward_list)

        max_check_point = max(max_reward_by_checkpoint.items(), key=lambda x: (x[1], x[0]))
        mean_check_point = max(mean_reward_by_checkpoint.items(), key=lambda x: (x[1], x[0]))
        if Parameters.log_on:
            logging.info(f"max_checkpoint: {max_check_point}")
            logging.info(f"mean_checkpoint: {mean_check_point}")
        print(f"max_checkpoint: {max_check_point}")
        print(f"mean_checkpoint: {mean_check_point}")
        print(f"Best Makespan: {best_makespan} with checkpoint: {best_checkpoint}")
        print("평가가 종료되었습니다.")

    @classmethod
    def script_performance(cls, env, n_epi, epsilon,memory, score, type, makespan_list, tardiness_list, score_list, pm_count, phase):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        if phase == "train":
            output_string = f"train_episode: {n_epi}, score: {score:.3f}, n_buffer: {len(memory)}, eps: {epsilon * 100:.1f}%, makespan: {makespan:.3f}, pm count: {pm_count}"
        else:
            output_string = f"test_episode: {n_epi}, score: {score:.3f}, n_buffer: {len(memory)}, eps: {epsilon * 100:.1f}%, makespan: {makespan:.3f}, pm count: {pm_count}"
        if type:
            makespan_list.append(makespan)
            tardiness_list.append(Total_Tardiness_time)
            score_list.append(score)
        if Parameters.log_on:
            print(output_string)
            logging.info(f'performance :{output_string}')
        return makespan_list, tardiness_list, score_list