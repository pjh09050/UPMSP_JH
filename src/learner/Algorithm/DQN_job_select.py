from src.learner.common.Qnet import *
from src.learner.common.ReplayBuffer import *
import torch.nn.functional as F
import torch.optim as optim
from src.simlator.Simulator import *
from matplotlib import pyplot as plt
from src.common.Parameters import *
import numpy as np
import re
np.set_printoptions(suppress=True)

class DQN_Job_Select:
    @classmethod
    def train(cls, q, q_target, memory, optimizer, loss_list):
        for i in range(10):
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)
            q_out = q(s)
            q_a = q_out.gather(1, a)

            # 가능한 액션의 인덱스 리스트
            valid_actions = Simulator.get_valid_actions(s_prime)
            q_prime_all = q_target(s_prime)
            q_prime_all[~valid_actions] = float('-inf')
            # 모든 값이 -inf라면 0으로 처리
            max_q_prime = torch.where(valid_actions.any(dim=1, keepdim=True), q_prime_all.max(1, keepdim=True)[0], torch.zeros_like(q_prime_all[:, :1]))

            # max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

            # # Online network에서 다음 상태의 행동 선택
            # max_a_prime = q(s_prime).max(1)[1].unsqueeze(1)  # 행동 선택 (index)
            # # Target network에서 선택한 행동의 Q값 추출
            # max_q_prime = q_target(s_prime).gather(1, max_a_prime)

            target = r + Hyperparameters.gamma * max_q_prime * (1 - done_mask)
            loss = F.smooth_l1_loss(q_a, target)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
            optimizer.step()

    @classmethod
    def train_n_step(cls, q, q_target, memory, optimizer, loss_list):
        for i in range(10):
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)
            q_out = q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

            # # Online network에서 다음 상태의 행동 선택
            # max_a_prime = q(s_prime).max(1)[1].unsqueeze(1)
            # max_q_prime = q_target(s_prime).gather(1, max_a_prime)

            target = r + (Hyperparameters.gamma ** memory.n_step) * max_q_prime * done_mask

            loss = F.smooth_l1_loss(q_a, target)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
            optimizer.step()

    @classmethod
    def main(cls):
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)

        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer(Hyperparameters.buffer_limit)
        # memory = ReplayBuffer_n_step(Hyperparameters.buffer_limit, n_step=10, gamma=Hyperparameters.gamma)

        optimizer = optim.Adam(q.parameters(), lr=Hyperparameters.learning_rate)

        makespan_list = []
        last_50_makespan = []
        last_50_score = []
        tardiness_list = []
        last_50_tardiness = []
        score_list = []
        loss_list = []

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)

        plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(plot_save_directory, exist_ok=True)

        for n_epi in range(Hyperparameters.episode):
            for dataid in Parameters.datasetId:
                # epsilon = max(0.2, 1 - n_epi / Hyperparameters.episode)
                epsilon = max(0, 1 - n_epi / Hyperparameters.episode)
                s = env.reset(dataid)
                done = False
                score = 0.0
                while not done:
                    # print('s', s)
                    a_list = q.sample_action_list(torch.from_numpy(s).float(), epsilon)
                    s_prime, r, done, a = env.job_select(a_list, epsilon, n_epi)
                    done_mask = 1.0 if done else 0.0
                    # print('a', a)
                    # print('r', r)
                    # print('s_prime', s_prime)
                    # print('done', done)
                    # print('done_mask', done_mask)
                    # print('------------------------------------------------------')
                    memory.put((s, a, r, s_prime, done_mask))
                    s = s_prime
                    score += r
                    if done:
                        break

            if n_epi % 10 == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())

            if memory.size() > 4000:
                cls.train(q, q_target, memory, optimizer, loss_list)
                makespan_list, tardiness_list, score_list = cls.script_performance(env, n_epi, epsilon, memory, score, True, makespan_list, tardiness_list, score_list)
                last_50_makespan.append(np.mean(makespan_list[-50:]))
                last_50_score.append(np.mean(score_list[-50:]))
                last_50_tardiness.append(np.mean(tardiness_list[-50:]))
                # cls.train_n_step(q, q_target, memory, optimizer, loss_list)

            # 결과 및 파라미터 저장
            if Parameters.param_down_on:
                params = q.state_dict()
                file_name = str(n_epi) + "param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(params, file_path)

        # 학습 및 테스트 결과를 2열 3행 subplot에 그리기
        object = Hyperparameters.object
        print('object:', object)
        if object == 'total_tardiness':
            object = 'Total Tardiness'
            train_data = tardiness_list
            last_train_50 = last_50_tardiness
        else:
            object = 'Makespan'
            train_data = makespan_list
            last_train_50 = last_50_makespan

        fig, axs = plt.subplots(3, 2, figsize=(58, 26))
        axs[0, 0].plot(range(len(train_data)), train_data, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 0].set_xlabel('Episodes', fontsize=36)
        axs[0, 0].set_ylabel(f'Training {object}', fontsize=36)
        axs[0, 0].tick_params(axis='both', labelsize=28)

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

        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red', label='Training Loss')
        axs[2, 0].set_xlabel('Episodes', fontsize=36)
        axs[2, 0].set_ylabel('Training Loss', fontsize=36)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        correlation = np.corrcoef(train_data, score_list)[0, 1]
        axs[2, 1].scatter(train_data, score_list, color='black', marker='o', label=f'Correlation: {correlation:.2f}')
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
        if Parameters.gantt_on_check:
            env.gantt_chart(score, None)

        plt.show()
        print("학습이 종료되었습니다.")

    @classmethod
    def get_result(cls, parameter, dataSets):
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        params = torch.load(parameter)
        q.load_state_dict(params)
        q.eval()
        episode = int(re.search(r'\\(\d+)param\.pt', parameter).group(1))
        for data_id in dataSets:
            s = env.reset(data_id)
            done = False
            score = 0.0
            while not done:
                print('s', s)
                epsilon = 0
                a, a_list = q.select_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done, a = env.job_select(a_list, epsilon, episode)
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
            env.gantt_chart(score, None)

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
    def get_evaluate(cls, checkpoint_path, number_of_checkpoint, point_num, datasets):
        env = Simulator
        file_list = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))]
        interver = len(file_list)//number_of_checkpoint
        check_point_list = [i for i in range(1, len(file_list)) if i%interver == 0 ]
        check_point_list.append(len(file_list)-1)

        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        mean_reward_by_checkpoint = {}
        max_reward_by_checkpoint = {}

        best_makespan = float('inf')
        best_checkpoint = None

        for check_point_number in check_point_list[-point_num:]:
            check_point = f"{checkpoint_path}/{check_point_number}param.pt"
            params = torch.load(check_point)
            q.load_state_dict(params)
            q.eval()
            reward_list = []
            for dataset in datasets:
                s = env.reset(dataset)
                done = False
                score = 0.0
                while not done:
                    epsilon = 0
                    a, a_list = q.select_action(torch.from_numpy(s).float(), epsilon)
                    s_prime, r, done, a = env.job_select(a_list, epsilon, check_point_number)
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

                reward_list.append(round(score,3))
                if Parameters.log_on:
                    logging.info(f"checkpoint: {check_point_number}")
                    logging.info(f"dataset: {dataset}")
                    logging.info(f"score:{round(score,3)}")

                # print(f"checkpoint: {check_point_number}")
                # print(f"dataset: {dataset}")
                # print(f"score:{score}")

            mean_reward_by_checkpoint[check_point_number] = sum(reward_list) / len(reward_list)
            max_reward_by_checkpoint[check_point_number] = max(reward_list)

        max_check_point = max(max_reward_by_checkpoint.items(), key=lambda x: (x[1], x[0]))
        mean_check_point = max(mean_reward_by_checkpoint.items(), key=lambda x: (x[1], x[0]))
        # max_check_point = max(max_reward_by_checkpoint.items(), key=lambda x: x[1])[0]
        # mean_check_point = max(mean_reward_by_checkpoint.items(), key=lambda x: x[1])[0]

        if Parameters.log_on:
            logging.info(f"max_checkpoint: {max_check_point}")
            logging.info(f"mean_checkpoint: {mean_check_point}")
        print(f"max_score_checkpoint: {max_check_point}")
        print(f"mean_checkpoint: {mean_check_point}")
        print(f"Best Makespan: {best_makespan} with checkpoint: {best_checkpoint}")
        print("평가가 종료되었습니다.")

    @classmethod
    def script_performance(cls, env, n_epi, epsilon,memory, score, type, makespan_list, tardiness_list, score_list):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        # output_string = f"n_episode: {n_epi}, score: {score:.3f}, n_buffer: {memory.size()}, eps: {epsilon * 100:.1f}%\n" + \
        #                 f"Total Flow time: {Total_Flow_time:.2f}, util: {util:.2f}, makespan: {makespan:.3f}\n" + \
        #                 f"Total Lateness: {Total_Lateness_time:.2f}, Total Tardiness: {Total_Tardiness_time:.2f}, T_max: {T_max:.2f}\n" + \
        #                 f"--------------------------------------------------------------------------------------"
        output_string = f"n_episode: {n_epi}, score: {score:.3f}, n_buffer: {memory.size()}, eps: {epsilon * 100:.1f}%, makespan: {makespan:.3f}"
        if type:
            makespan_list.append(makespan)
            tardiness_list.append(Total_Tardiness_time)
            score_list.append(score)
        if Parameters.log_on:
            # if n_epi % 50 == 0:
            print(output_string)
            logging.info(f'performance :{output_string}')
        return makespan_list, tardiness_list, score_list








