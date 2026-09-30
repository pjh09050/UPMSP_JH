import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from src.simlator.Simulator import *
from src.common.Parameters import *
import re

class MonteCarloNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MonteCarloNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Linear(16, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

class MC_Job_Select:
    batch_size = 128

    @classmethod
    def train(cls, model, optimizer, memory, loss_list):
        if len(memory) < cls.batch_size:
            return

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

            # # Reward Scaling 적용
            # returns = np.array(targets)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # target_tensor = torch.tensor(returns, dtype=torch.float32)

            state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            action_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
            target_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor).squeeze(0).numpy()
        return q_values

    @classmethod
    def main(cls):
        env = Simulator()
        model = MonteCarloNN(Hyperparameters.input_layer, Hyperparameters.output_layer)
        optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.learning_rate)
        memory = deque(maxlen=200)

        makespan_list = []
        last_50_makespan = []
        last_50_score = []
        loss_list = []
        score_list = []

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)

        plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(plot_save_directory, exist_ok=True)

        for n_epi in range(Hyperparameters.episode):
            for dataid in Parameters.datasetId:
                s = env.reset(dataid)
                done = False
                score = 0.0
                episode = []
                epsilon = max(0, 1.0 - n_epi / Hyperparameters.episode)
                while not done:
                    # print('s', s)
                    a_list = cls.get_action_list(s, model)
                    s_prime, r, done, a = env.mc_job_select(a_list, epsilon, n_epi)
                    episode.append((s, a, r))
                    # print('a', a)
                    # print('r', r)
                    # print('s_prime', s_prime)
                    # print('done', done)
                    # print('------------------------------------------------------')
                    s = s_prime
                    score += r
                    if done:
                        break

                memory.append(episode)
                cls.train(model, optimizer, memory, loss_list)
                makespan_list, score_list = cls.script_performance(env, n_epi, epsilon, memory, score, True, makespan_list, score_list)
                last_50_makespan.append(np.mean(makespan_list[-50:]))
                last_50_score.append(np.mean(score_list[-50:]))

            if Parameters.param_down_on:
                file_name = str(n_epi) + "param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(model.state_dict(), file_path)

        # Plot results
        object = 'Makespan'
        fig, axs = plt.subplots(3, 2, figsize=(58, 26))
        # 학습 목적함수 (왼쪽 첫 번째)
        axs[0, 0].plot(range(len(makespan_list)), makespan_list, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 0].set_xlabel('Episodes', fontsize=36)
        axs[0, 0].set_ylabel(f'Training {object}', fontsize=36)
        axs[0, 0].tick_params(axis='both', labelsize=28)
        # 학습 Score (왼쪽 두 번째)
        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='blue', label='Training Score')
        axs[1, 0].set_xlabel('Episodes', fontsize=36)
        axs[1, 0].set_ylabel('Training Score', fontsize=36)
        axs[1, 0].tick_params(axis='both', labelsize=28)
        # 최근 50개 makespan 평균 (오른쪽 첫 번째)
        axs[0, 1].plot(range(len(last_50_makespan)), last_50_makespan, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 1].set_xlabel('Episodes', fontsize=36)
        axs[0, 1].set_ylabel(f'Average {object} over previous 50 episodes', fontsize=36)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        axs[1, 1].plot(range(len(last_50_score)), last_50_score, linestyle='--', marker='.', color='blue', label=f'Training Score')
        axs[1, 1].set_xlabel('Episodes', fontsize=36)
        axs[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=36)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        # 학습 Loss (오른쪽 두 번째)
        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red', label='Training Loss')
        axs[2, 0].set_xlabel('Episodes', fontsize=36)
        axs[2, 0].set_ylabel('Training Loss', fontsize=36)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        # 테스트 Score, 목적함수 상관관계 (오른쪽 세 번째)
        correlation = np.corrcoef(makespan_list, score_list)[0, 1]
        axs[2, 1].scatter(makespan_list, score_list, color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        axs[2, 1].set_xlabel(f'{object}', fontsize=36)
        axs[2, 1].set_ylabel('Score', fontsize=36)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        axs[2, 1].legend(fontsize=28)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name_1 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_train_result.png"
            file_path_1 = os.path.join(plot_save_directory, result_name_1)
            plt.savefig(file_path_1)
            print(f"첫 번째 그래프가 저장되었습니다: {file_path_1}")
        plt.show()

        if Parameters.gantt_on_check:
            env.gantt_chart(score, None)
        print("학습 및 테스트가 종료되었습니다.")

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
                    a_list = cls.get_action_list(s, model)
                    s_prime, r, done, a = env.mc_job_select(a_list, epsilon, check_point_number)
                    s = s_prime
                    score += r
                    if done:
                        break

                Flow_time, util, makespan, tardiness, lateness, t_max = env.performance_measure()

                if makespan <= best_makespan:
                    best_makespan = makespan
                    best_checkpoint = check_point_number

                    print(f"checkpoint: {check_point_number}")
                    print(f"dataset: {dataset}")
                    print(f"score:{score}")

                reward_list.append(round(score, 3))
                if Parameters.log_on:
                    logging.info(f"checkpoint: {check_point_number}")
                    logging.info(f"dataset: {dataset}")
                    logging.info(f"score:{round(score, 3)}")

                # print(f"checkpoint: {check_point_number}")
                # print(f"dataset: {dataset}")
                # print(f"score:{score}")

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
    def get_result(cls, parameter, dataSets):
        env = Simulator
        model = MonteCarloNN(Hyperparameters.input_layer, Hyperparameters.output_layer)
        params = torch.load(parameter)
        model.load_state_dict(params)
        model.eval()
        episode = int(re.search(r'\\(\d+)param\.pt', parameter).group(1))
        for data_id in dataSets:
            s = env.reset(data_id)
            done = False
            score = 0.0
            while not done:
                epsilon = 0
                print('s', s)
                a_list = cls.get_action_list(s, model)
                print('a_list', a_list)
                s_prime, r, done, a = env.mc_job_select(a_list, epsilon, episode)
                print('max_a', np.argmax(a_list))
                print('a', a)
                print('r', r)
                print('s_prime', s_prime)
                print('done', done)
                print('------------------------------------------------------')
                s = s_prime
                score += r
                if done:
                    break

            Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, t_max = env.performance_measure()
            env.gantt_chart(score)

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
            print("평가가 종료되었습니다.")

    @classmethod
    def script_performance(cls, env, n_epi, epsilon, memory, score, type, makespan_list, score_list):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        output_string = f"n_episode: {n_epi}, n_buffer: {len(memory)}, eps: {epsilon * 100:.1f}%, score: {score:.2f}, makespan: {makespan:.2f}\n" + \
                        f"--------------------------------------------------------------------"
        if type:
            makespan_list.append(makespan)
            score_list.append(score)
        if Parameters.log_on:
            print(output_string)
            logging.info(f'performance :{output_string}')
        return makespan_list, score_list
