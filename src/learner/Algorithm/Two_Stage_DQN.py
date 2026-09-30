from src.learner.common.ReplayBuffer import *
from src.simlator.Simulator import *
from matplotlib import pyplot as plt
from src.common.Parameters import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(suppress=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Qnet(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(Qnet, self).__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.model = nn.Sequential(
            nn.Linear(input_layer, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, output_layer)
        )

    def forward(self, x):
        x = x.to(device)
        return self.model(x)

    def sample_action_list(self, obs):
        obs = obs.to(device)
        return self.forward(obs)

    def sample_action(self, obs, epsilon):
        obs = obs.to(device)
        out = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0, self.output_layer - 1)
        return out.argmax().item()

    def select_action(self, obs):
        obs = obs.to(device)
        return self.forward(obs).argmax().item()

class Two_Stage_DQN:
    q_pm = None
    q_pm_target = None
    optimizer_pm = None
    memory_pm = None
    makespan_list_pm, tardiness_list_pm, score_list_pm, loss_list_pm = [], [], [], []
    last_50_makespan_pm, last_50_tardiness_pm, last_50_score_pm = [], [], []

    q_job = None
    q_job_target = None
    optimizer_job = None
    memory_job = None
    makespan_list_job, tardiness_list_job, score_list_job, loss_list_job = [], [], [], []
    last_50_makespan_job, last_50_tardiness_job, last_50_score_job = [], [], []

    makespan_list_pm_finetune, tardiness_list_pm_finetune, score_list_pm_finetune, loss_list_pm_finetune = [], [], [], []
    last_50_makespan_pm_finetune, last_50_tardiness_pm_finetune, last_50_score_pm_finetune = [], [], []

    number_of_checkpoint = 4000
    point_num = 1500

    @classmethod
    def init(cls):
        cls.env = Simulator()

        cls.number_of_checkpoint = Hyperparameters.episode
        cls.point_num = Hyperparameters.episode

        cls.q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        cls.q_pm_target = Qnet(Hyperparameters.input_layer, 2).to(device)
        cls.q_pm_target.load_state_dict(cls.q_pm.state_dict())
        cls.optimizer_pm = optim.Adam(cls.q_pm.parameters(), lr=Hyperparameters.learning_rate)
        cls.memory_pm = ReplayBuffer(int(Hyperparameters.buffer_limit * 1.5))

        cls.q_job = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        cls.q_job_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        cls.q_job_target.load_state_dict(cls.q_job.state_dict())
        cls.optimizer_job = optim.Adam(cls.q_job.parameters(), lr=Hyperparameters.learning_rate)
        cls.memory_job = ReplayBuffer(int(Hyperparameters.buffer_limit))

        cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm, cls.loss_list_pm = [], [], [], []
        cls.last_50_makespan_pm, cls.last_50_tardiness_pm, cls.last_50_score_pm = [], [], []

        cls.makespan_list_job, cls.tardiness_list_job, cls.score_list_job, cls.loss_list_job = [], [], [], []
        cls.last_50_makespan_job, cls.last_50_tardiness_job, cls.last_50_score_job = [], [], []

        cls.makespan_list_pm_finetune, cls.tardiness_list_pm_finetune, cls.score_list_pm_finetune, cls.loss_list_pm_finetune = [], [], [], []
        cls.last_50_makespan_pm_finetune, cls.last_50_tardiness_pm_finetune, cls.last_50_score_pm_finetune = [], [], []

    @classmethod
    def first_train(cls, q, q_target, memory, optimizer, loss_list):
        for i in range(20):
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)
            s, a, r, s_prime, done_mask = s.to(device), a.to(device), r.to(device), s_prime.to(device), done_mask.to(device)
            q_out = q(s)
            q_a = q_out.gather(1, a)

            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)

            target = r + Hyperparameters.gamma * max_q_prime * (1 - done_mask)
            loss = F.smooth_l1_loss(q_a, target)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
            optimizer.step()

    @classmethod
    def second_train(cls, q, q_target, memory, optimizer, loss_list):
        for i in range(20):
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)
            s, a, r, s_prime, done_mask = s.to(device), a.to(device), r.to(device), s_prime.to(device), done_mask.to(device)
            q_out = q(s)
            q_a = q_out.gather(1, a)

            valid_actions = Simulator.get_valid_two_step_actions(s_prime).to(device)
            q_prime_all = q_target(s_prime)
            q_prime_all[~valid_actions] = float('-inf')
            max_q_prime = torch.where(valid_actions.any(dim=1, keepdim=True), q_prime_all.max(1, keepdim=True)[0], torch.zeros_like(q_prime_all[:, :1]))

            target = r + Hyperparameters.gamma * max_q_prime * (1 - done_mask)
            loss = F.smooth_l1_loss(q_a, target)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
            optimizer.step()

    @classmethod
    def main(cls):
        cls.init()

        save_directory_agent1 = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}_agent1"
        save_directory_agent2 = os.path.join(pathConfig.model_save_path,f"{Parameters.simulation_time}_{Parameters.datasetId}_agent2")
        if Parameters.param_down_on:
            os.makedirs(save_directory_agent1, exist_ok=True)
            os.makedirs(save_directory_agent2, exist_ok=True)

        plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(plot_save_directory, exist_ok=True)

        # === Agent 1 학습 ===
        cls.train_agent1(save_directory_agent1)
        cls.plot_results(plot_save_directory, cls.loss_list_pm, 'pm')
        print("Agent 1 학습이 종료되었습니다.")
        # === Agent 1 평가 ===
        # best_checkpoint_pm = cls.evaluate_agent1(save_directory)
        best_checkpoint_pm = cls.number_of_checkpoint - 1
        # === Agent 2 학습 ===
        cls.train_agent2(save_directory_agent1, save_directory_agent2, best_checkpoint_pm)
        cls.plot_results(plot_save_directory, cls.loss_list_job, 'family')
        print("Agent 2 학습이 종료되었습니다.")
        # === Agent 2 평가 ===
        best_checkpoint_family = cls.evaluate_agent2(save_directory_agent1, best_checkpoint_pm, save_directory_agent2, plot_save_directory)
        # === Agent 2 결과 ===
        cls.result_agent2(save_directory_agent1, best_checkpoint_pm, save_directory_agent2, best_checkpoint_family)
        # === Agent 1 Fine Tuning ===
        # cls.fine_tuning_agent1(save_directory_finetune)
        # cls.plot_results(plot_save_directory, cls.loss_list_pm_finetune, 'pm_finetune')
        # print("Agent 1 Fine-tuning이 종료되었습니다.")

    @classmethod
    def train_agent1(cls, save_directory):
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / Hyperparameters.episode)
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                while not done:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a = cls.q_pm.sample_action(s_tensor, epsilon)
                    s_prime, r, done, _ = cls.env.two_step_select(a, epsilon, 'first')
                    done_mask = 1.0 if done else 0.0
                    cls.memory_pm.put((s, a, r, s_prime, done_mask))
                    s = s_prime
                    score += r
                    if done:
                        break

            if cls.memory_pm.size() > Hyperparameters.buffer_limit / 10:
                cls.first_train(cls.q_pm, cls.q_pm_target, cls.memory_pm, cls.optimizer_pm, cls.loss_list_pm)
                cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm = cls.script_performance(cls.env, n_epi, epsilon, cls.memory_pm, score, cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm)
                cls.last_50_makespan_pm.append(np.mean(cls.makespan_list_pm[-50:]))
                cls.last_50_score_pm.append(np.mean(cls.score_list_pm[-50:]))
                cls.last_50_tardiness_pm.append(np.mean(cls.tardiness_list_pm[-50:]))

            if n_epi % 20 == 0:
                cls.q_pm_target.load_state_dict(cls.q_pm.state_dict())

            if Parameters.param_down_on:
                params = cls.q_pm.state_dict()
                file_name = str(n_epi) + "pm_param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(params, file_path)

    @classmethod
    def evaluate_agent1(cls, save_directory):
        file_list = [f for f in os.listdir(save_directory) if os.path.isfile(os.path.join(save_directory, f))]
        file_list = [f for f in file_list if 'pm_param.pt' in f]
        file_list = sorted(file_list, key=lambda x: int(x.replace("pm_param.pt", "")))

        datasets = Parameters.datasetId

        interver = len(file_list) // cls.number_of_checkpoint
        check_point_list = [i for i in range(1, len(file_list)) if i % interver == 0]
        check_point_list.append(len(file_list) - 1)

        q = Qnet(Hyperparameters.input_layer, 2).to(device)

        best_checkpoint = None
        best_score = float('-inf')
        best_makespan = float('inf')
        best_pm_count = 0

        for check_point_number in check_point_list[-cls.point_num:]:
            check_point = f"{save_directory}/{check_point_number}pm_param.pt"
            params = torch.load(check_point)
            q.load_state_dict(params)
            q.eval()
            pm_count = 0
            skip_checkpoint = False

            for dataset in datasets:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0
                while not done:
                    epsilon = 0
                    a = q.sample_action(torch.from_numpy(s).float(), epsilon)
                    if a == 0:
                        pm_count += 1
                    s_prime, r, done, _ = cls.env.two_step_select(a, epsilon, 'first')
                    s = s_prime
                    score += r
                    if pm_count >= Simulator.number_of_job:
                        skip_checkpoint = True
                        break
                    if done:
                        break

                if skip_checkpoint:
                    print(f"Checkpoint {check_point_number} skipped due to excessive PM.")
                    break  # dataset 루프 탈출

                Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
                if best_score <= score:
                    best_score = score
                    best_checkpoint = check_point_number
                    best_makespan = makespan
                    best_pm_count = pm_count

                    if Parameters.log_on:
                        print(f"checkpoint: {best_checkpoint}")
                        print(f"dataset: {dataset}")
                        print(f"score: {best_score}")
                        print(f"makespan: {best_makespan}")
                        print(f'pm_count: {best_pm_count}')

            if skip_checkpoint:
                continue  # checkpoint 루프로 다시 이동

        print(f'best checkpoint: {best_checkpoint}')
        print(f"best_score: {best_score}")
        print(f'best_makespan: {best_makespan}')
        print(f'best_pm_count: {best_pm_count}')
        print("Agent 1 평가가 종료되었습니다.")

        # 최고 성능 모델 경로 저장
        best_path = os.path.join(save_directory, f"{best_checkpoint}pm_param.pt")
        torch.save({'best_path': best_path}, os.path.join(save_directory, f"{best_checkpoint}_best_pm_info.pt"))
        return best_checkpoint

    @classmethod
    def train_agent2(cls, save_directory, save_directory_agent2, best_checkpoint):
        # best_info_path = os.path.join(save_directory, f"{best_checkpoint}_best_pm_info.pt")
        # if os.path.exists(best_info_path):
        #     best_info = torch.load(best_info_path)
        #     best_param_path = best_info['best_path']
        #     print(f"[Agent2] Best Agent1 model loaded from: {best_param_path}")
        #     cls.q_pm.load_state_dict(torch.load(best_param_path))
        # else:
        #     raise FileNotFoundError(f"[Agent2] best_pm_info.pt not found in: {save_directory}")

        best_info_path = os.path.join(save_directory, f"{best_checkpoint}pm_param.pt")
        print(f"[Agent2] Best Agent1 model loaded from: {best_info_path}")
        cls.q_pm.load_state_dict(torch.load(best_info_path, map_location=device))
        cls.q_pm.to(device)
        cls.q_pm.eval()
        pm_count = 0
        score = 0.0
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / Hyperparameters.episode)
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                pm_count = 0
                while not done:
                    with torch.no_grad(): # Agent 1은 고정된 상태
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_pm = cls.q_pm.select_action(s_tensor)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, is_pm = cls.env.two_step_select(a_pm, epsilon, 'first')
                        score += r_pm
                        s = s_pm
                        if pm_count >= Simulator.number_of_job:
                            break
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = cls.q_job.sample_action_list(s_tensor)
                        s_prime, r_job, done, a = cls.env.two_step_select(a_list, epsilon, 'second')
                        done_mask = 1.0 if done else 0.0
                        cls.memory_job.put((s, a, r_job, s_prime, done_mask))
                        s = s_prime
                        score += r_job
                    if done:
                        break

            if pm_count >= Simulator.number_of_job:
                print(f"[Agent2] Episode {n_epi} skipped due to excessive PMs ({pm_count})")
                continue  # Skip training on this episode

            if cls.memory_job.size() > Hyperparameters.buffer_limit / 10:
                cls.second_train(cls.q_job, cls.q_job_target, cls.memory_job, cls.optimizer_job, cls.loss_list_job)
                cls.makespan_list_job, cls.tardiness_list_job, cls.score_list_job = cls.script_performance(cls.env, n_epi, epsilon, cls.memory_job, score, cls.makespan_list_job, cls.tardiness_list_job, cls.score_list_job)
                cls.last_50_makespan_job.append(np.mean(cls.makespan_list_job[-50:]))
                cls.last_50_score_job.append(np.mean(cls.score_list_job[-50:]))
                cls.last_50_tardiness_job.append(np.mean(cls.tardiness_list_job[-50:]))

            if n_epi % 20 == 0:
                cls.q_job_target.load_state_dict(cls.q_job.state_dict())

            if Parameters.param_down_on:
                params = cls.q_job.state_dict()
                file_name = str(n_epi) + "family_param.pt"
                file_path = os.path.join(save_directory_agent2, file_name)
                torch.save(params, file_path)

    @classmethod
    def evaluate_agent2(cls, save_directory_agent1, best_checkpoint_pm, save_directory_agent2, plot_save_directory):
        file_list = [f for f in os.listdir(save_directory_agent2) if os.path.isfile(os.path.join(save_directory_agent2, f))]
        file_list = [f for f in file_list if 'family_param.pt' in f]
        file_list = sorted(file_list, key=lambda x: int(x.replace("family_param.pt", "")))

        datasets = Parameters.datasetId

        interver = len(file_list) // cls.number_of_checkpoint if len(file_list) >= cls.number_of_checkpoint else 1
        check_point_list = [i for i in range(1, len(file_list)) if i % interver == 0]
        check_point_list.append(len(file_list) - 1)

        best_pm_info_path = os.path.join(save_directory_agent1, f"{best_checkpoint_pm}pm_param.pt")
        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_pm_info_path, map_location=device))
        q_pm.eval()

        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)

        best_checkpoint = None
        best_score = float('-inf')
        best_makespan = float('inf')
        best_pm_count = 0

        evaluation_scores = []  # 체크포인트별 점수 저장용
        for check_point_number in check_point_list[-cls.point_num:]:
            check_point_path = os.path.join(save_directory_agent2, f"{check_point_number}family_param.pt")
            if not os.path.exists(check_point_path):
                print(f"[Skip] {check_point_path} not found.")
                continue
            try:
                params = torch.load(check_point_path, map_location=device)
                q_family.load_state_dict(params)
                q_family.eval()
            except Exception as e:
                print(f"[Error] Failed to load checkpoint {check_point_path}: {e}")
                continue

            pm_count = 0
            skip_checkpoint = False
            total_score = 0.0

            for dataset in datasets:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0

                while not done:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_pm = q_pm.select_action(s_tensor)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, _ = cls.env.two_step_select(a_pm, 0, 'first')
                        s = s_pm
                        score += r_pm
                        if pm_count >= Simulator.number_of_job:
                            skip_checkpoint = True
                            break
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = q_family.sample_action_list(s_tensor)
                        s_prime, r_job, done, a = cls.env.two_step_select(a_list, 0, 'second')
                        s = s_prime
                        score += r_job

                    if done:
                        break

                total_score += score

                if skip_checkpoint:
                    print(f"Checkpoint {check_point_number} skipped due to excessive PM.")
                    break

            evaluation_scores.append(total_score)

            if not skip_checkpoint:
                Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
                if best_score <= total_score:
                    best_score = total_score
                    best_checkpoint = check_point_number
                    best_makespan = makespan
                    best_pm_count = pm_count
                    if Parameters.log_on:
                        print(f"checkpoint: {best_checkpoint}")
                        print(f"score: {best_score}")
                        print(f"makespan: {best_makespan}")
                        print(f'pm_count: {best_pm_count}')
                else:
                    if Parameters.log_on:
                        print(f"checkpoint: {check_point_number}")

        # 그래프 그리기 (skip된 에피소드 시각화 포함)
        if evaluation_scores:
            # full_scores = [None] * cls.number_of_checkpoint
            # for ep, score in evaluation_scores:
            #     if 0 <= ep < cls.number_of_checkpoint:
            #         full_scores[ep] = score
            # x = list(range(1, cls.number_of_checkpoint + 1))
            plt.figure(figsize=(24, 12))
            plt.plot(range(len(evaluation_scores)), evaluation_scores, marker='.', linestyle='--', color='blue')
            plt.xlabel("Episodes", fontsize=36)
            plt.ylabel("Cumulative Reward", fontsize=36)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.tight_layout()
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_agent2_evaluation_scores.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(os.path.join(file_path))
            plt.show()
        print(f'[Evaluate Agent2] best checkpoint: {best_checkpoint}')
        print(f'[Evaluate Agent2] best_score: {best_score}')
        print(f'[Evaluate Agent2] best_makespan: {best_makespan}')
        print(f'[Evaluate Agent2] best_pm_count: {best_pm_count}')
        print("Agent 2 평가가 종료되었습니다.")

        best_path = os.path.join(save_directory_agent2, f"{best_checkpoint}family_param.pt")
        torch.save({'best_path': best_path}, os.path.join(save_directory_agent2, f"{best_checkpoint}_best_family_info.pt"))
        return best_checkpoint

    @classmethod
    def result_agent2(cls, save_directory_agent1, best_checkpoint_pm, save_directory_agent2, best_checkpoint_family):
        # PM 네트워크 로딩
        # best_pm_info_path = os.path.join(save_directory_agent1, f"{best_checkpoint_pm}_best_pm_info.pt")
        # best_pm_info = torch.load(best_pm_info_path)
        # best_pm_path = best_pm_info['best_path']
        # q_pm = Qnet(Hyperparameters.input_layer, 2)
        # q_pm.load_state_dict(torch.load(best_pm_path))
        # q_pm.eval()

        best_info_path = os.path.join(save_directory_agent1, f"{best_checkpoint_pm}pm_param.pt")
        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_info_path, map_location=device))
        q_pm.eval()

        # Family 네트워크 로딩
        best_family_info_path = os.path.join(save_directory_agent2, f"{best_checkpoint_family}_best_family_info.pt")
        best_family_info = torch.load(best_family_info_path)
        best_family_path = best_family_info['best_path']
        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        q_family.load_state_dict(torch.load(best_family_path, map_location=device))
        q_family.eval()

        print(f"[Result] agent1: {best_checkpoint_pm}, agent2: {best_checkpoint_family} pt 최종 평가 시작")
        for data_id in Parameters.datasetId:
            s = cls.env.reset(data_id)
            done = False
            score = 0.0
            pm_count = 0
            while not done:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a_pm = q_pm.select_action(s_tensor)
                if a_pm == 0:
                    pm_count += 1
                    s_pm, r_pm, done, _ = cls.env.two_step_select(a_pm, 0, 'first')
                    s = s_pm
                    score += r_pm
                else:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a_list = q_family.sample_action_list(s_tensor)
                    s_prime, r_job, done, a = cls.env.two_step_select(a_list, 0, 'second')
                    s = s_prime
                    score += r_job

            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            cls.env.gantt_chart(score, f'{best_checkpoint_family}result_agent2')
            print(f"[Result] Dataset: {data_id}")
            print(f"[Result] Score: {score}")
            print(f"[Result] Makespan: {round(makespan, 3)}")

    @classmethod
    def fine_tuning_agent1(cls, save_directory):
        # === Agent 1 Fine-tuning (with Agent 2 fixed) ===
        print("[Agent2] Fine-tuning Agent 1 based on updated environment")
        cls.memory_pm = ReplayBuffer(int(Hyperparameters.buffer_limit * 1.5))
        for n_epi in range(int(Hyperparameters.episode / 2)):
            epsilon = 0.1
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                while not done:
                    a = cls.q_pm.sample_action(torch.from_numpy(s).float(), epsilon)
                    s_prime, r_pm, done, _ = cls.env.two_step_select(a, epsilon, 'first')
                    done_mask = 1.0 if done else 0.0
                    cls.memory_pm.put((s, a, r_pm, s_prime, done_mask))
                    s = s_prime
                    score += r_pm
                    if done:
                        break

            if cls.memory_pm.size() > Hyperparameters.batch_size:
                cls.first_train(cls.q_pm, cls.q_pm_target, cls.memory_pm, cls.optimizer_pm, cls.loss_list_pm_finetune)
                cls.makespan_list_pm_finetune, cls.tardiness_list_pm_finetune, cls.score_list_pm_finetune = cls.script_performance(cls.env, n_epi, epsilon, cls.memory_pm, score, cls.makespan_list_pm_finetune, cls.tardiness_list_pm_finetune, cls.score_list_pm_finetune)
                cls.last_50_makespan_pm_finetune.append(np.mean(cls.makespan_list_pm_finetune[-50:]))
                cls.last_50_score_pm_finetune.append(np.mean(cls.score_list_pm_finetune[-50:]))
                cls.last_50_tardiness_pm_finetune.append(np.mean(cls.tardiness_list_pm_finetune[-50:]))

            if n_epi % 10 == 0:
                cls.q_pm_target.load_state_dict(cls.q_pm.state_dict())

            if Parameters.param_down_on:
                params = cls.q_job.state_dict()
                file_name = str(n_epi) + "fine_tune_param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(params, file_path)

    @classmethod
    def get_result(cls, save_directory_agent1, save_directory_agent2, datasets):
        cls.init()
        # === best_pm_info.pt 자동 탐색 ===
        best_pm_info_file = next((f for f in os.listdir(save_directory_agent1) if f.endswith("_best_pm_info.pt")), None)
        if best_pm_info_file is None:
            raise FileNotFoundError(f"No _best_pm_info.pt file found in {save_directory_agent1}")
        best_pm_info_path = os.path.join(save_directory_agent1, best_pm_info_file)
        best_pm_info = torch.load(best_pm_info_path)
        best_pm_path = best_pm_info['best_path']
        best_checkpoint_pm = best_pm_info_file.split('_')[0]

        # === best_family_info.pt 자동 탐색 ===
        best_family_info_file = next((f for f in os.listdir(save_directory_agent2) if f.endswith("_best_family_info.pt")), None)
        if best_family_info_file is None:
            raise FileNotFoundError(f"No _best_family_info.pt file found in {save_directory_agent2}")
        best_family_info_path = os.path.join(save_directory_agent2, best_family_info_file)
        best_family_info = torch.load(best_family_info_path)
        best_family_path = best_family_info['best_path']
        best_checkpoint_family = best_family_info_file.split('_')[0]

        q_pm = Qnet(Hyperparameters.input_layer, 2)
        q_pm.load_state_dict(torch.load(best_pm_path))
        q_pm.eval()

        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_family.load_state_dict(torch.load(best_family_path))
        q_family.eval()

        print(f"[Result] agent1: {best_checkpoint_pm}, agent2: {best_checkpoint_family} pt - {datasets} 데이터셋 평가 시작")
        for data_id in datasets:
            s = cls.env.reset(data_id)
            done = False
            score = 0.0
            pm_count = 0

            while not done:
                with torch.no_grad():
                    a_pm = q_pm.select_action(torch.from_numpy(s).float())

                if a_pm == 0:
                    pm_count += 1
                    s_pm, r_pm, done, _ = cls.env.two_step_select(a_pm, 0, 'first')
                    s = s_pm
                    score += r_pm
                else:
                    a_list = q_family.sample_action_list(torch.from_numpy(s).float())
                    s_prime, r_job, done, a = cls.env.two_step_select(a_list, 0, 'second')
                    s = s_prime
                    score += r_job

            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            # 파일 경로에서 마지막 [] 안의 텍스트 추출
            match = re.search(r"\[([^\[\]]+)\]", save_directory_agent1.replace("\\", "/"))
            dataset_label = match.group(1) if match else "unknown"
            cls.env.gantt_chart(score, f'training_dataset_{dataset_label}')

            print(f"[Result] Dataset: {data_id}")
            print(f"[Result] Score: {score}")
            print(f"[Result] Makespan: {round(makespan, 3)}")

    @classmethod
    def script_performance(cls, env, n_epi, epsilon,memory, score, makespan_list, tardiness_list, score_list):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        output_string = f"n_episode: {n_epi}, score: {score:.3f}, n_buffer: {memory.size()}, eps: {epsilon * 100:.1f}%, makespan: {makespan:.3f}"
        makespan_list.append(makespan)
        tardiness_list.append(Total_Tardiness_time)
        score_list.append(score)
        if Parameters.log_on:
            # if n_epi % 50 == 0:
            print(output_string)
            logging.info(f'performance :{output_string}')
        return makespan_list, tardiness_list, score_list

    @classmethod
    def plot_results(cls, plot_save_directory, loss_list, agent_name):
        object = Hyperparameters.object
        print(f'object: {object}')

        if agent_name == 'pm':
            score_list = cls.score_list_pm
            tardiness_list = cls.tardiness_list_pm
            last_50_score = cls.last_50_score_pm
            last_50_tardiness = cls.last_50_tardiness_pm
            makespan_list = cls.makespan_list_pm
            last_50_makespan = cls.last_50_makespan_pm
        elif agent_name == 'family':
            score_list = cls.score_list_job
            tardiness_list = cls.tardiness_list_job
            last_50_score = cls.last_50_score_job
            last_50_tardiness = cls.last_50_tardiness_job
            makespan_list = cls.makespan_list_job
            last_50_makespan = cls.last_50_makespan_job
        elif agent_name == 'pm_finetune':
            score_list = cls.score_list_pm_finetune
            tardiness_list = cls.tardiness_list_pm_finetune
            last_50_score = cls.last_50_score_pm_finetune
            last_50_tardiness = cls.last_50_tardiness_pm_finetune
            makespan_list = cls.makespan_list_pm_finetune
            last_50_makespan = cls.last_50_makespan_pm_finetune
        else:
            raise ValueError(f"Invalid agent_name: {agent_name}")

        if object == 'total_tardiness':
            object_title = 'Total Tardiness'
            train_data = tardiness_list
            last_train_50 = last_50_tardiness
        else:
            object_title = 'Makespan'
            train_data = makespan_list
            last_train_50 = last_50_makespan

        fig, axs = plt.subplots(3, 2, figsize=(58, 26))
        axs[0, 0].plot(range(len(train_data)), train_data, linestyle='--', marker='.', color='green')
        axs[0, 0].set_xlabel('Episodes', fontsize=36)
        axs[0, 0].set_ylabel(f'Training {object_title}', fontsize=36)
        axs[0, 0].tick_params(axis='both', labelsize=28)

        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='blue')
        axs[1, 0].set_xlabel('Episodes', fontsize=36)
        axs[1, 0].set_ylabel('Training Score', fontsize=36)
        axs[1, 0].tick_params(axis='both', labelsize=28)

        axs[0, 1].plot(range(len(last_train_50)), last_train_50, linestyle='--', marker='.', color='green')
        axs[0, 1].set_xlabel('Episodes', fontsize=36)
        axs[0, 1].set_ylabel(f'Average {object_title} over previous 50 episodes', fontsize=36)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        axs[1, 1].plot(range(len(last_50_score)), last_50_score, linestyle='--', marker='.', color='blue')
        axs[1, 1].set_xlabel('Episodes', fontsize=36)
        axs[1, 1].set_ylabel('Average Score over previous 50 episodes', fontsize=36)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red')
        axs[2, 0].set_xlabel('Episodes', fontsize=36)
        axs[2, 0].set_ylabel('Training Loss', fontsize=36)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        correlation = np.corrcoef(train_data, score_list)[0, 1] if len(train_data) > 1 and len(score_list) > 1 else 0
        axs[2, 1].scatter(train_data, score_list, color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        axs[2, 1].set_xlabel(f'{object_title}', fontsize=36)
        axs[2, 1].set_ylabel('Score', fontsize=36)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        axs[2, 1].legend(fontsize=28)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_{agent_name}_result.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path)
            print(f"{agent_name} 학습 그래프 저장 완료: {file_path}")

        if Parameters.gantt_on_check and agent_name != "pm":
            cls.env.gantt_chart(score_list[-1], agent_name)

        plt.show()
