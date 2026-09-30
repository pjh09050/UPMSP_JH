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

    def sample_action(self, obs, epsilon, pm_end):
        obs = obs.to(device)
        out = self.forward(obs)
        if pm_end == True:
            return 1
        else:
            if random.random() < epsilon:
                return random.randint(0, self.output_layer - 1)
            return out.argmax().item()

    def select_action(self, obs):
        obs = obs.to(device)
        return self.forward(obs).argmax().item()

class MCReplayBuffer:
    def __init__(self, max_episodes=1000):
        self.episodes = []
        self.current_episode = []
        self.max_episodes = max_episodes

    def put(self, transition):
        self.current_episode.append(transition)

    def end_episode(self):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
        self.episodes.append(self.current_episode)
        self.current_episode = []

    def get_all_episodes(self):
        return self.episodes

    def clear(self):
        self.episodes = []

    def size(self):
        return len(self.episodes)

class Two_Stage_MC_DQN_2:
    q_pm = None
    optimizer_pm = None
    memory_pm = None
    score_list_pm = [np.nan] * Hyperparameters.episode
    makespan_list_pm = [np.nan] * Hyperparameters.episode
    tardiness_list_pm = [np.nan] * Hyperparameters.episode
    loss_list_pm = [np.nan] * Hyperparameters.episode
    last_50_score_pm = [np.nan] * Hyperparameters.episode
    last_50_makespan_pm = [np.nan] * Hyperparameters.episode
    last_50_tardiness_pm = [np.nan] * Hyperparameters.episode

    q_job = None
    q_job_target = None
    optimizer_job = None
    memory_job = None
    makespan_list_job = [np.nan] * Hyperparameters.episode
    tardiness_list_job = [np.nan] * Hyperparameters.episode
    score_list_job = [np.nan] * Hyperparameters.episode
    loss_list_job = [np.nan] * Hyperparameters.episode
    last_50_makespan_job = [np.nan] * Hyperparameters.episode
    last_50_tardiness_job = [np.nan] * Hyperparameters.episode
    last_50_score_job = [np.nan] * Hyperparameters.episode

    makespan_list_pm_finetune = [np.nan] * Hyperparameters.episode
    tardiness_list_pm_finetune = [np.nan] * Hyperparameters.episode
    score_list_pm_finetune = [np.nan] * Hyperparameters.episode
    loss_list_pm_finetune = [np.nan] * Hyperparameters.episode
    last_50_makespan_pm_finetune = [np.nan] * Hyperparameters.episode
    last_50_tardiness_pm_finetune = [np.nan] * Hyperparameters.episode
    last_50_score_pm_finetune = [np.nan] * Hyperparameters.episode

    mc_batch_size = 128
    mc_buffer_limit = 200
    mc_learning_rate = 0.0001
    number_of_checkpoint = 2000
    point_num = 1500
    sim = None

    @classmethod
    def init(cls):
        cls.env = Simulator()

        cls.mc_batch_size = 64
        cls.mc_buffer_limit = 64
        cls.mc_learning_rate = 0.0001
        cls.number_of_checkpoint = Hyperparameters.episode
        cls.point_num = Hyperparameters.episode

        cls.q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        cls.optimizer_pm = optim.Adam(cls.q_pm.parameters(), lr=cls.mc_learning_rate)
        cls.memory_pm = MCReplayBuffer(max_episodes=cls.mc_buffer_limit)

        cls.q_job = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        cls.q_job_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        cls.q_job_target.load_state_dict(cls.q_job.state_dict())
        cls.optimizer_job = optim.Adam(cls.q_job.parameters(), lr=Hyperparameters.learning_rate)
        cls.memory_job = ReplayBuffer(int(Hyperparameters.buffer_limit))

        cls.score_list_pm = [np.nan] * Hyperparameters.episode
        cls.makespan_list_pm = [np.nan] * Hyperparameters.episode
        cls.tardiness_list_pm = [np.nan] * Hyperparameters.episode
        cls.loss_list_pm = [np.nan] * Hyperparameters.episode
        cls.last_50_score_pm = [np.nan] * Hyperparameters.episode
        cls.last_50_makespan_pm = [np.nan] * Hyperparameters.episode
        cls.last_50_tardiness_pm = [np.nan] * Hyperparameters.episode

        cls.makespan_list_job = [np.nan] * Hyperparameters.episode
        cls.tardiness_list_job = [np.nan] * Hyperparameters.episode
        cls.score_list_job = [np.nan] * Hyperparameters.episode
        cls.loss_list_job = [np.nan] * Hyperparameters.episode
        cls.last_50_makespan_job = [np.nan] * Hyperparameters.episode
        cls.last_50_tardiness_job = [np.nan] * Hyperparameters.episode
        cls.last_50_score_job = [np.nan] * Hyperparameters.episode

        cls.makespan_list_pm_finetune = [np.nan] * Hyperparameters.episode
        cls.tardiness_list_pm_finetune = [np.nan] * Hyperparameters.episode
        cls.score_list_pm_finetune = [np.nan] * Hyperparameters.episode
        cls.loss_list_pm_finetune = [np.nan] * Hyperparameters.episode
        cls.last_50_makespan_pm_finetune = [np.nan] * Hyperparameters.episode
        cls.last_50_tardiness_pm_finetune = [np.nan] * Hyperparameters.episode
        cls.last_50_score_pm_finetune = [np.nan] * Hyperparameters.episode

        cls.rule_ratio_history = []

    @classmethod
    def get_state_feature_names(cls):
        """
        Table 2 기준으로 state feature 이름 정의.
        실제 state 구성 순서와 반드시 맞춰야 함.
        필요하면 이 리스트를 네 state vector 순서에 맞게 수정할 것.
        """
        nf = Hyperparameters.family_type

        feature_names = []

        # Waiting jobs
        for f in range(1, nf + 1):
            feature_names.append(f"num_wait_f{f}")
        for f in range(1, nf + 1):
            feature_names.append(f"avg_p_f{f}")
        for f in range(1, nf + 1):
            feature_names.append(f"avg_d_f{f}")

        # Idle machine setup one-hot
        for f in range(1, nf + 1):
            feature_names.append(f"setup_f{f}")

        # Machine-related scalar features
        feature_names += [
            "num_pm",
            "time_since_last_pm",
            "machine_quality",
            "accum_deterioration",
            "num_setups",
            "accum_setup_time",
        ]

        return feature_names

    @classmethod
    def collect_explainability_trace(cls, q_pm, q_job, dataset_ids):
        """
        최종 policy 기준으로 PMC/JA trace 수집
        PMC: every decision point
        JA: PM이 아닌 경우 rule selection이 발생한 state만 저장
        """
        pm_states, pm_actions = [], []
        ja_states, ja_actions = [], []

        q_pm.eval()
        q_job.eval()

        for data_id in dataset_ids:
            s = cls.env.reset(data_id)
            done = False
            pm_count = 0
            pm_end = False

            while not done:
                s_tensor = torch.from_numpy(s).float().to(device)

                if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                    pm_end = True

                with torch.no_grad():
                    a_pm = q_pm.sample_action(s_tensor, 0, pm_end)

                # PMC trace 저장
                pm_states.append(np.array(s, copy=True))
                pm_actions.append(a_pm)

                if a_pm == 0:  # PM 수행
                    pm_count += 1
                    s_prime, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                    s = s_prime
                else:
                    with torch.no_grad():
                        a_list = q_job.sample_action_list(s_tensor)

                    s_prime, r_job, done, a_job, _ = cls.env.two_step_select(a_list, 0, 'second')

                    # JA trace 저장
                    ja_states.append(np.array(s, copy=True))
                    ja_actions.append(a_job)

                    s = s_prime

        return {
            "pm_states": np.array(pm_states),
            "pm_actions": np.array(pm_actions),
            "ja_states": np.array(ja_states),
            "ja_actions": np.array(ja_actions),
        }

    @classmethod
    def compute_pmc_saliency(cls, q_pm, states):
        """
        PMC saliency:
        delta Q = Q(PM) - Q(NoPM)
        네 코드에서 a==0 이 PM으로 동작하므로 action index 0 = PM, 1 = No PM 가정
        """
        if len(states) == 0:
            return None, None

        q_pm.eval()
        x = torch.tensor(states, dtype=torch.float32, device=device, requires_grad=True)

        q = q_pm(x)  # shape: (N, 2)
        delta_q = (q[:, 0] - q[:, 1]).sum()

        q_pm.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        delta_q.backward()

        saliency = x.grad.detach().abs().cpu().numpy()
        saliency_mean = saliency.mean(axis=0)
        saliency_mean = saliency_mean / (saliency_mean.sum() + 1e-8)

        return saliency_mean, saliency

    @classmethod
    def compute_ja_saliency(cls, q_job, states, actions):
        """
        JA saliency:
        chosen Q(s, a*)에 대한 input gradient
        """
        if len(states) == 0:
            return None, None

        q_job.eval()
        x = torch.tensor(states, dtype=torch.float32, device=device, requires_grad=True)
        a = torch.tensor(actions, dtype=torch.long, device=device)

        q = q_job(x)
        chosen_q = q.gather(1, a.view(-1, 1)).sum()

        q_job.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        chosen_q.backward()

        saliency = x.grad.detach().abs().cpu().numpy()
        saliency_mean = saliency.mean(axis=0)
        saliency_mean = saliency_mean / (saliency_mean.sum() + 1e-8)

        return saliency_mean, saliency

    @classmethod
    def get_pmc_action_conditioned_stats(cls, states, actions):
        """
        PMC: PM vs No PM 상태 평균 비교
        """
        if len(states) == 0:
            return None

        states = np.asarray(states)
        actions = np.asarray(actions)

        pm_states = states[actions == 0]
        nopm_states = states[actions == 1]

        stats = {
            "pm_count": len(pm_states),
            "nopm_count": len(nopm_states),
            "pm_mean": pm_states.mean(axis=0) if len(pm_states) > 0 else None,
            "nopm_mean": nopm_states.mean(axis=0) if len(nopm_states) > 0 else None,
        }
        return stats

    @classmethod
    def get_ja_action_conditioned_stats(cls, states, actions, num_actions):
        """
        JA: rule(action)별 평균 state 계산
        """
        if len(states) == 0:
            return None

        states = np.asarray(states)
        actions = np.asarray(actions)

        out = {}
        for a in range(num_actions):
            group = states[actions == a]
            out[a] = {
                "count": len(group),
                "mean": group.mean(axis=0) if len(group) > 0 else None
            }
        return out

    @classmethod
    def plot_feature_importance_bar(cls, importances, feature_names, title, save_path, top_k=10):
        if importances is None:
            return

        idx = np.argsort(importances)[::-1][:top_k]
        sorted_imp = importances[idx]
        sorted_names = [feature_names[i] for i in idx]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_imp)), sorted_imp, edgecolor='black')
        plt.xticks(range(len(sorted_imp)), sorted_names, rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel("Normalized saliency", fontsize=14)
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @classmethod
    def plot_pmc_action_conditioned_bar(cls, stats, feature_names, save_path, top_k=8):
        """
        PM 선택 시점 vs No PM 선택 시점 평균 state 비교
        차이가 큰 feature 위주로 bar plot
        """
        if stats is None or stats["pm_mean"] is None or stats["nopm_mean"] is None:
            return

        pm_mean = stats["pm_mean"]
        nopm_mean = stats["nopm_mean"]

        diff = np.abs(pm_mean - nopm_mean)
        idx = np.argsort(diff)[::-1][:top_k]

        x = np.arange(len(idx))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, pm_mean[idx], width, label="PM selected", edgecolor='black')
        plt.bar(x + width/2, nopm_mean[idx], width, label="No PM", edgecolor='black')

        plt.xticks(x, [feature_names[i] for i in idx], rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel("Average feature value", fontsize=14)
        plt.title("PMC action-conditioned state statistics", fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @classmethod
    def plot_ja_rule_conditioned_heatmap(cls, stats, feature_names, save_path, top_k=10):
        """
        JA rule별 평균 state를 heatmap으로 표시
        """
        if stats is None:
            return

        valid_actions = [a for a, v in stats.items() if v["mean"] is not None and v["count"] > 0]
        if len(valid_actions) == 0:
            return

        mean_matrix = np.array([stats[a]["mean"] for a in valid_actions])  # (num_rules, state_dim)

        # feature variance 기준으로 informative한 feature만 선택
        feat_var = np.var(mean_matrix, axis=0)
        idx = np.argsort(feat_var)[::-1][:top_k]

        selected_matrix = mean_matrix[:, idx]
        selected_feature_names = [feature_names[i] for i in idx]

        # row label
        rule_names = []
        for a in valid_actions:
            family_idx = a // 4 + 1
            rule_idx = a % 4
            rule_map = ["SPTSDR", "SPTLDR", "LPTSDR", "LPTLDR"]
            rule_names.append(f"F{family_idx}_{rule_map[rule_idx]}")

        plt.figure(figsize=(12, 6))
        plt.imshow(selected_matrix, aspect='auto', cmap='YlGnBu')
        plt.colorbar()
        plt.xticks(range(len(selected_feature_names)), selected_feature_names, rotation=45, ha='right', fontsize=11)
        plt.yticks(range(len(rule_names)), rule_names, fontsize=11)
        plt.title("JA rule-conditioned state statistics", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @classmethod
    def save_explainability_tables(cls, pm_stats, ja_stats, feature_names, save_dir):
        """
        CSV 저장
        """
        os.makedirs(save_dir, exist_ok=True)

        # PMC
        if pm_stats is not None and pm_stats["pm_mean"] is not None and pm_stats["nopm_mean"] is not None:
            pm_df = pd.DataFrame({
                "feature": feature_names,
                "pm_mean": pm_stats["pm_mean"],
                "nopm_mean": pm_stats["nopm_mean"],
                "abs_diff": np.abs(pm_stats["pm_mean"] - pm_stats["nopm_mean"])
            })
            pm_df.to_csv(os.path.join(save_dir, "pm_action_conditioned_stats.csv"), index=False)

        # JA
        if ja_stats is not None:
            rows = []
            for a, info in ja_stats.items():
                if info["mean"] is None:
                    continue
                for fname, val in zip(feature_names, info["mean"]):
                    rows.append({
                        "action": a,
                        "feature": fname,
                        "mean_value": val,
                        "count": info["count"]
                    })
            if len(rows) > 0:
                ja_df = pd.DataFrame(rows)
                ja_df.to_csv(os.path.join(save_dir, "ja_action_conditioned_stats.csv"), index=False)

    @classmethod
    def explain_stage1_stage2_agents(cls, q_pm, q_job, dataset_ids, save_dir):
        """
        설명력 분석 메인 함수
        """
        os.makedirs(save_dir, exist_ok=True)

        trace = cls.collect_explainability_trace(q_pm, q_job, dataset_ids)

        pm_states = trace["pm_states"]
        pm_actions = trace["pm_actions"]
        ja_states = trace["ja_states"]
        ja_actions = trace["ja_actions"]

        feature_names = cls.get_state_feature_names()

        # -------- PMC saliency --------
        pm_saliency_mean, _ = cls.compute_pmc_saliency(q_pm, pm_states)
        if pm_saliency_mean is not None:
            cls.plot_feature_importance_bar(
                pm_saliency_mean,
                feature_names,
                "PMC feature importance (gradient-based saliency)",
                os.path.join(save_dir, "pmc_saliency_bar.png"),
                top_k=10
            )
            pd.DataFrame({
                "feature": feature_names,
                "importance": pm_saliency_mean
            }).sort_values("importance", ascending=False).to_csv(
                os.path.join(save_dir, "pmc_saliency.csv"), index=False
            )

        # -------- JA saliency --------
        ja_saliency_mean, _ = cls.compute_ja_saliency(q_job, ja_states, ja_actions)
        if ja_saliency_mean is not None:
            cls.plot_feature_importance_bar(
                ja_saliency_mean,
                feature_names,
                "JA feature importance (gradient-based saliency)",
                os.path.join(save_dir, "ja_saliency_bar.png"),
                top_k=10
            )
            pd.DataFrame({
                "feature": feature_names,
                "importance": ja_saliency_mean
            }).sort_values("importance", ascending=False).to_csv(
                os.path.join(save_dir, "ja_saliency.csv"), index=False
            )

        # -------- PMC action-conditioned stats --------
        pm_stats = cls.get_pmc_action_conditioned_stats(pm_states, pm_actions)
        cls.plot_pmc_action_conditioned_bar(
            pm_stats,
            feature_names,
            os.path.join(save_dir, "pmc_action_conditioned_bar.png"),
            top_k=8
        )

        # -------- JA action-conditioned stats --------
        ja_stats = cls.get_ja_action_conditioned_stats(
            ja_states,
            ja_actions,
            Hyperparameters.output_layer
        )
        cls.plot_ja_rule_conditioned_heatmap(
            ja_stats,
            feature_names,
            os.path.join(save_dir, "ja_rule_conditioned_heatmap.png"),
            top_k=10
        )

        # -------- save CSV --------
        cls.save_explainability_tables(pm_stats, ja_stats, feature_names, save_dir)

        print(f"[Explainability] saved to: {save_dir}")

    @classmethod
    def first_mc_train(cls, q, memory, optimizer, loss_list):
        if memory.size() < cls.mc_batch_size:
            return
        batch = random.sample(memory.get_all_episodes(), cls.mc_batch_size)
        for episode in batch:
            G = 0
            states, actions, targets = [], [], []
            for state, action, reward, _, _ in reversed(episode):
                G = reward + Hyperparameters.gamma * G
                states.append(state)
                actions.append(action)
                targets.append(G)

            states.reverse()
            actions.reverse()
            targets.reverse()

            # Reward Scaling 적용
            # returns = np.array(targets)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # target_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

            state_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            action_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
            target_tensor = torch.tensor(np.array(targets), dtype=torch.float32).to(device)

            q_values = q(state_tensor).gather(1, action_tensor).squeeze(1)
            loss = nn.HuberLoss()(q_values, target_tensor.detach())

            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)
            optimizer.step()

    @classmethod
    def second_train(cls, q, q_target, memory, optimizer, loss_list):
        for _ in range(20):
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
            torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=0.5)
            optimizer.step()

    @classmethod
    def main(cls):
        cls.init()

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}_agent1"
        save_directory_agent2 = os.path.join(pathConfig.model_save_path,f"{Parameters.simulation_time}_{Parameters.datasetId}_agent2")
        # save_directory_finetune = os.path.join(pathConfig.model_save_path,f"{Parameters.simulation_time}_{Parameters.datasetId}_finetune")

        # save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.datasetId}_agent1"
        # save_directory_agent2 = os.path.join(pathConfig.model_save_path,f"{Parameters.datasetId}_agent2")
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)
            os.makedirs(save_directory_agent2, exist_ok=True)
            # os.makedirs(save_directory_finetune, exist_ok=True)

        cls.plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(cls.plot_save_directory, exist_ok=True)

        cls.running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(cls.running_data_save_directory, exist_ok=True)

        # === Agent 1 학습 ===
        train_start_time = time.time()
        cls.train_agent1(save_directory)
        cls.plot_results(cls.plot_save_directory, cls.loss_list_pm, 'pm')
        print("Agent 1 학습이 종료되었습니다.")
        # === Agent 1 평가 ===
        # best_checkpoint_pm = cls.evaluate_agent1(save_directory)
        best_checkpoint_pm = cls.number_of_checkpoint - 1
        # === Agent 2 결과 ===
        # cls.result_agent1(save_directory, best_checkpoint_pm)
        # === Agent 2 학습 ===
        cls.train_agent2(save_directory, save_directory_agent2, best_checkpoint_pm)
        cls.plot_results(cls.plot_save_directory, cls.loss_list_job, 'family')
        print("Agent 2 학습이 종료되었습니다.")
        # === Agent 2 평가 ===
        best_checkpoint_family = cls.evaluate_agent2(save_directory, best_checkpoint_pm, save_directory_agent2, cls.plot_save_directory)
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f"model train time: {train_elapsed_time:.4f} sec")
        # === Agent 2 결과 ===
        makespan, eval_time = cls.result_agent2(save_directory, best_checkpoint_pm, save_directory_agent2, best_checkpoint_family, cls.plot_save_directory)
        txt_name = (
            f"{Parameters.simulation_time}_"
            f"{Hyperparameters.object}_"
            f"{Hyperparameters.action_type}_"
            f"{Parameters.datasetId}_"
            f"F{Hyperparameters.family_type}_"
            f"agent2_second_stage_best_result.txt"
        )
        txt_path = os.path.join(cls.running_data_save_directory, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("===== Second Stage Result =====\n")
            f.write(f"training Time (sec): {train_elapsed_time:.6f}\n")
            f.write(f"Evaluation Time (sec): {eval_time:.6f}\n")
        print(f"📄 Best result TXT 저장 완료: {txt_path}")
        return makespan
        # === Agent 1 Fine Tuning ===
        # cls.fine_tuning_agent1(save_directory_finetune)
        # cls.plot_results(plot_save_directory, cls.loss_list_pm_finetune, 'pm_finetune')
        # print("Agent 1 Fine-tuning이 종료되었습니다.")

    @classmethod
    def get_simulator_instance(cls):
        if cls.sim is None:
            from src.simlator.Simulator import Simulator
            cls.sim = Simulator()
        return cls.sim

    @classmethod
    def train_agent1(cls, save_directory):
        score = 0.0
        pm_count = 0
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / (Hyperparameters.episode - 1))
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                pm_count = 0
                pm_end = False
                while not done:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_end = True
                    a = cls.q_pm.sample_action(s_tensor, epsilon, pm_end)
                    if a == 0:
                        pm_count += 1
                    s_prime, r, done, _, _ = cls.env.two_step_select(a, epsilon, 'first')
                    cls.memory_pm.put((s, a, r, s_prime, done))
                    s = s_prime
                    score += r
                    if done:
                        if pm_end == True:
                            print(f"Training Agent1 - Episode {n_epi} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break
                cls.memory_pm.end_episode()

            cls.first_mc_train(cls.q_pm, cls.memory_pm, cls.optimizer_pm, cls.loss_list_pm)
            cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm = cls.script_performance(cls.env, n_epi, epsilon, cls.memory_pm, score, cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm, pm_count)
            cls.last_50_score_pm[n_epi] = (
                np.nanmean(cls.score_list_pm[max(0, n_epi - 49):n_epi + 1])
                if np.any(~np.isnan(cls.score_list_pm[max(0, n_epi - 49):n_epi + 1])) else np.nan
            )
            cls.last_50_makespan_pm[n_epi] = (
                np.nanmean(cls.makespan_list_pm[max(0, n_epi - 49):n_epi + 1])
                if np.any(~np.isnan(cls.makespan_list_pm[max(0, n_epi - 49):n_epi + 1])) else np.nan
            )
            cls.last_50_tardiness_pm[n_epi] = (
                np.nanmean(cls.tardiness_list_pm[max(0, n_epi - 49):n_epi + 1])
                if np.any(~np.isnan(cls.tardiness_list_pm[max(0, n_epi - 49):n_epi + 1])) else np.nan
            )
            if Parameters.param_down_on:
                params = cls.q_pm.state_dict()
                file_name = str(n_epi) + "pm_param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(params, file_path)

        cls.env.gantt_chart(score, f"result_agent1")
        print(f"[Agent1 Result] Dataset: {Parameters.datasetId}")
        print(f"[Agent1 Result] Score: {score}")
        print(f"[Agent1 Result] Makespan: {round(cls.makespan_list_pm[-1], 3)}")

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

        for check_point_number in check_point_list:
            check_point = f"{save_directory}/{check_point_number}pm_param.pt"
            params = torch.load(check_point, map_location=device)
            q.load_state_dict(params)
            q.eval()
            pm_count = 0
            skip_checkpoint = False
            for dataset in datasets:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0
                while not done:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a = q.select_action(s_tensor)
                    if a == 0:
                        pm_count += 1
                    s_prime, r, done, _, _ = cls.env.two_step_select(a, 0, 'first')
                    s = s_prime
                    score += r
                    if pm_count >= Simulator.number_of_job:
                        skip_checkpoint = True
                        break
                    if done:
                        break

                if skip_checkpoint:
                    print(f"Checkpoint {check_point_number} skipped due to excessive PM.")
                    break

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

        print(f'[Evaluate Agent1] best checkpoint: {best_checkpoint}')
        print(f"[Evaluate Agent1] best_score: {best_score}")
        print(f'[Evaluate Agent1] best_makespan: {best_makespan}')
        print(f'[Evaluate Agent1] best_pm_count: {best_pm_count}')
        print("Agent 1 평가가 종료되었습니다.")

        # 최고 성능 모델 경로 저장
        best_path = os.path.join(save_directory, f"{best_checkpoint}pm_param.pt")
        torch.save({'best_path': best_path}, os.path.join(save_directory, f"{best_checkpoint}_best_pm_info.pt"))
        return best_checkpoint

    @classmethod
    def result_agent1(cls, save_directory, best_checkpoint_pm):
        # # best_pm_info 불러오기
        # best_pm_info_path = os.path.join(save_directory, f"{best_checkpoint_pm}_best_pm_info.pt")
        # if not os.path.exists(best_pm_info_path):
        #     raise FileNotFoundError(f"[Evaluate+Result] best_pm_info.pt not found at {best_pm_info_path}")
        #
        # best_pm_info = torch.load(best_pm_info_path)
        # best_pm_path = best_pm_info['best_path']

        pm_files = [f for f in os.listdir(save_directory) if f.endswith("pm_param.pt")]
        pm_files = sorted(pm_files, key=lambda x: int(x.replace("pm_param.pt", "")))
        last_pm_file = pm_files[-1]
        best_pm_path = os.path.join(save_directory, last_pm_file)

        # Q-network 로딩
        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_pm_path))
        q_pm.eval()

        print(f"[Result - Agent1 Only] best_checkpoint: {best_checkpoint_pm}")
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
                s_prime, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                s = s_prime
                score += r_pm

            # 결과 출력
            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            cls.env.gantt_chart(score, f"result_agent1")
            print(f"[Agent1 Result] Dataset: {data_id}")
            print(f"[Agent1 Result] Score: {score}")
            print(f"[Agent1 Result] Makespan: {round(makespan, 3)}")

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

        # cls.rule_ratio_history = []  # [{rule: ratio, ...}, episode index 순서]
        score = 0.0
        pm_count = 0
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / (Hyperparameters.episode - 1))
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                pm_count = 0
                pm_end = False
                while not done:
                    with torch.no_grad(): # Agent 1은 고정된 상태
                        s_tensor = torch.from_numpy(s).float().to(device)
                        if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                            pm_end = True
                        a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, epsilon, 'first')
                        s = s_pm
                        score += r_pm
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = cls.q_job.sample_action_list(s_tensor)
                        s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, epsilon, 'second')
                        done_mask = 1.0 if done else 0.0
                        cls.memory_job.put((s, a, r_job, s_prime, done_mask))
                        s = s_prime
                        score += r_job
                    if done:
                        if pm_end == True:
                            print(f"Training Agent2 - Episode {n_epi} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break

            if cls.memory_job.size() > Hyperparameters.buffer_limit / 10:
                cls.second_train(cls.q_job, cls.q_job_target, cls.memory_job, cls.optimizer_job, cls.loss_list_job)
                cls.makespan_list_job, cls.tardiness_list_job, cls.score_list_job = cls.script_performance(cls.env, n_epi, epsilon, cls.memory_job, score, cls.makespan_list_job, cls.tardiness_list_job, cls.score_list_job, pm_count)
                cls.last_50_score_job[n_epi] = (
                    np.nanmean(cls.score_list_job[max(0, n_epi - 49):n_epi + 1])
                    if np.any(~np.isnan(cls.score_list_job[max(0, n_epi - 49):n_epi + 1])) else np.nan
                )
                cls.last_50_makespan_job[n_epi] = (
                    np.nanmean(cls.makespan_list_job[max(0, n_epi - 49):n_epi + 1])
                    if np.any(~np.isnan(cls.makespan_list_job[max(0, n_epi - 49):n_epi + 1])) else np.nan
                )
                cls.last_50_tardiness_job[n_epi] = (
                    np.nanmean(cls.tardiness_list_job[max(0, n_epi - 49):n_epi + 1])
                    if np.any(~np.isnan(cls.tardiness_list_job[max(0, n_epi - 49):n_epi + 1])) else np.nan
                )
                if Parameters.param_down_on:
                    params = cls.q_job.state_dict()
                    file_name = str(n_epi) + "family_param.pt"
                    file_path = os.path.join(save_directory_agent2, file_name)
                    torch.save(params, file_path)
            if n_epi % 20 == 0:
                cls.q_job_target.load_state_dict(cls.q_job.state_dict())

    @classmethod
    def evaluate_agent2(cls, save_directory_agent1, best_checkpoint_pm, save_directory_agent2, plot_save_directory):
        file_list = [f for f in os.listdir(save_directory_agent2) if os.path.isfile(os.path.join(save_directory_agent2, f))]
        file_list = [f for f in file_list if 'family_param.pt' in f]
        check_point_list = sorted([int(f.replace("family_param.pt", "")) for f in file_list])

        # interver = len(file_list) // cls.number_of_checkpoint if len(file_list) >= cls.number_of_checkpoint else 1
        # check_point_list = [i for i in range(1, len(file_list)) if i % interver == 0]
        # check_point_list.append(len(file_list) - 1)

        # Load best Agent1 (PM) model
        # best_pm_info_path = os.path.join(save_directory_agent1, f"{best_checkpoint_pm}_best_pm_info.pt")
        # best_pm_info = torch.load(best_pm_info_path)
        # best_pm_path = best_pm_info['best_path']
        # q_pm = Qnet(Hyperparameters.input_layer, 2)
        # q_pm.load_state_dict(torch.load(best_pm_path))
        # q_pm.eval()

        best_pm_info_path = os.path.join(save_directory_agent1, f"{best_checkpoint_pm}pm_param.pt")
        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_pm_info_path, map_location=device))
        q_pm.eval()

        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)

        best_checkpoint = None
        best_score = float('-inf')
        best_makespan = float('inf')
        best_pm_count = 0

        evaluation_scores = {}  # checkpoint: score
        evaluation_makespans = {}
        for check_point_number in check_point_list:
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
            total_score = 0.0
            for dataset in Parameters.datasetId:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0
                pm_end = False
                while not done:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(s).float().to(device)
                        if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                            pm_end = True
                        a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                        s = s_pm
                        score += r_pm
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = q_family.sample_action_list(s_tensor)
                        s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, 0, 'second')
                        s = s_prime
                        score += r_job
                    if done:
                        if pm_end == True:
                            print(f"Training Agent2 - Episode {check_point_number} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break

                total_score += score
                total_score = np.round(total_score, decimals=3)

            evaluation_scores[check_point_number] = total_score
            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            evaluation_makespans[check_point_number] = makespan
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

        # 전체 checkpoint 기준으로 x축 맞추기
        all_checkpoints = list(range(Hyperparameters.episode))
        y_scores = [evaluation_scores.get(cp, np.nan) for cp in all_checkpoints]
        makespan_scores = [evaluation_makespans.get(cp, np.nan) for cp in all_checkpoints]

        if evaluation_scores:
            plt.figure(figsize=(20, 12))
            plt.plot(all_checkpoints, y_scores, marker='.', linestyle='--', color='black')
            plt.xlabel("The cumulative number of episodes", fontsize=36)
            plt.ylabel("Cumulative Reward", fontsize=36)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.tight_layout()
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_agent2_evaluation_scores.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(os.path.join(file_path))
            # plt.show()

        if evaluation_makespans:
            plt.figure(figsize=(20, 12))
            plt.plot(all_checkpoints, makespan_scores, marker='.', linestyle='--', color='black')
            plt.xlabel("The cumulative number of episodes", fontsize=36)
            plt.ylabel("Makespan", fontsize=36)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.tight_layout()
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_agent2_evaluation_makespan.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(os.path.join(file_path))
            # plt.show()

        print(f'[Evaluate Agent2] best checkpoint: {best_checkpoint}')
        print(f'[Evaluate Agent2] best_score: {best_score}')
        print(f'[Evaluate Agent2] best_makespan: {best_makespan}')
        print(f'[Evaluate Agent2] best_pm_count: {best_pm_count}')
        print("Agent 2 평가가 종료되었습니다.")

        best_path = os.path.join(save_directory_agent2, f"{best_checkpoint}family_param.pt")
        torch.save({'best_path': best_path}, os.path.join(save_directory_agent2, f"{best_checkpoint}_best_family_info.pt"))

        eval_df = pd.DataFrame({
            "episode": all_checkpoints,
            "evaluation_score": y_scores,
            "evaluation_makespan": makespan_scores,
        })
        csv_name = (
            f"{Parameters.simulation_time}_"
            f"{Hyperparameters.object}_"
            f"{Hyperparameters.action_type}_"
            f"{Parameters.datasetId}_"
            f"agent2_evaluation_scores_makespan.csv"
        )
        csv_path = os.path.join(cls.running_data_save_directory, csv_name)
        eval_df.to_csv(csv_path, index=False)
        print(f"Agent2 evaluation score makespan CSV 저장 완료: {csv_path}")
        return best_checkpoint

    @classmethod
    def result_agent2(cls, save_directory_agent1, best_checkpoint_pm, save_directory_agent2, best_checkpoint_family, plot_save_directory):
        # PM 네트워크 로딩
        # best_pm_info_path = os.path.join(save_directory_agent1, f"{best_checkpoint_pm}_best_pm_info.pt")
        # best_pm_info = torch.load(best_pm_info_path)
        # best_pm_path = best_pm_info['best_path']
        # q_pm = Qnet(Hyperparameters.input_layer, 2)
        # q_pm.load_state_dict(torch.load(best_pm_path))
        # q_pm.eval()
        start_time = time.time()

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
            pm_end = False
            while not done:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s).float().to(device)
                    if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_end = True
                    a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                if a_pm == 0:
                    pm_count += 1
                    s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                    s = s_pm
                    score += r_pm
                else:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a_list = q_family.sample_action_list(s_tensor)
                    s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, 0, 'second')
                    s = s_prime
                    score += r_job

            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            elapsed_min = (time.time() - start_time) / 60
            print(f"one episode (소요 시간: {elapsed_min:.2f}분)")
            cls.env.gantt_chart(score, f'{best_checkpoint_family}result_agent2')
            episode_ratio = cls.env.get_rule_ratio_dist()  # {'SPTDR':0.62,'LPTDR':0.38,...}
            cls.plot_rule_ratio_barplot(episode_ratio, plot_save_directory)
            cls.env.plot_rule_heatmap(plot_save_directory)
            cls.env.write_reward_txt(makespan, score, plot_save_directory)

            explain_dir = os.path.join(plot_save_directory, f"{Parameters.simulation_time}_explainability")
            cls.explain_stage1_stage2_agents(
                q_pm,
                q_family,
                Parameters.datasetId,
                explain_dir
            )

            print(f"[Result] Dataset: {data_id}")
            print(f"[Result] Score: {score}")
            print(f"[Result] Makespan: {round(makespan, 3)}")
            print(f"[Result] Rule ratio: {episode_ratio}")

            return round(makespan, 3), elapsed_min

    @classmethod
    def get_evaluate(cls, save_directory_agent1, save_directory_agent2, dataset_base):
        cls.init()
        # if isinstance(dataset_base, list):
        #     dataset_base = dataset_base[0]
        # if re.search(r'_U\d+$', dataset_base):
        #     dataset_base = dataset_base.rsplit('_U', 1)[0]
        #     datasets = [f"{dataset_base}_U{i}" for i in range(1, 11)]
        # elif re.search(r'_\d+$', dataset_base):
        #     dataset_base = dataset_base.rsplit('_', 1)[0]
        #     datasets = [f"{dataset_base}_{i}" for i in range(1, 11)]
        # else:
        #     raise ValueError("dataset_base should end with _U# or _#")
        pm_files = [f for f in os.listdir(save_directory_agent1) if f.endswith("pm_param.pt")]
        pm_files = sorted(pm_files, key=lambda x: int(x.replace("pm_param.pt", "")))
        last_pm_file = pm_files[-1]
        last_pm_path = os.path.join(save_directory_agent1, last_pm_file)

        # file_list = [f for f in os.listdir(save_directory_agent2) if os.path.isfile(os.path.join(save_directory_agent2, f))]
        # file_list = [f for f in file_list if 'family_param.pt' in f]
        # file_list = sorted(file_list, key=lambda x: int(x.replace("family_param.pt", "")))
        # interver = len(file_list) // cls.number_of_checkpoint if len(file_list) >= cls.number_of_checkpoint else 1
        # check_point_list = [i for i in range(1, len(file_list)) if i % interver == 0]
        # check_point_list.append(len(file_list) - 1)

        file_list = [f for f in os.listdir(save_directory_agent2) if 'family_param.pt' in f]
        checkpoint_numbers = sorted([int(f.replace("family_param.pt", "")) for f in file_list])
        interver = len(checkpoint_numbers) // cls.number_of_checkpoint if len(checkpoint_numbers) >= cls.number_of_checkpoint else 1
        check_point_list = [ckpt for i, ckpt in enumerate(checkpoint_numbers) if i % interver == 0]
        check_point_list.append(checkpoint_numbers[-1])  # 마지막도 포함

        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(last_pm_path, map_location=device))
        q_pm.eval()
        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)

        all_dataset_scores = {dataset: [] for dataset in dataset_base}
        all_dataset_makespans = {dataset: [] for dataset in dataset_base}
        avg_scores = []
        avg_makespans = []
        best_score_list = []
        existing_checkpoints = []

        for check_point_number in check_point_list:
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

            dataset_scores = []
            dataset_makespans = []
            for dataset in dataset_base:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0
                pm_count = 0
                pm_end = False
                while not done:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(s).float().to(device)
                        if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                            pm_end = True
                        a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                        s = s_pm
                        score += r_pm
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = q_family.sample_action_list(s_tensor)
                        s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, 0, 'second')
                        s = s_prime
                        score += r_job
                    if done:
                        if pm_end == True:
                            print(f"Training Agent2 - Episode {check_point_number} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break

                Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
                dataset_scores.append(np.round(score, decimals=3))
                dataset_makespans.append(np.round(makespan, decimals=3))
                all_dataset_scores[dataset].append(np.round(score, decimals=3))
                all_dataset_makespans[dataset].append(np.round(makespan, decimals=3))
                print(f"[Checkpoint {check_point_number}] Dataset: {dataset}")
                print(f"  → Score: {round(score, 3)}, Makespan: {round(makespan, 3)}")

                avg_scores.append(np.mean(dataset_scores))
                avg_makespans.append(np.mean(dataset_makespans))
                best_score_list.append(dataset_scores)
                existing_checkpoints.append(check_point_number)

        best_idx = int(np.argmax(avg_scores))
        best_checkpoint = existing_checkpoints[best_idx]
        best_scores = best_score_list[best_idx] if best_idx < len(best_score_list) else []
        score_mean = round(np.mean(best_scores), 3)
        score_std = round(np.std(best_scores), 3)

        train_dataset = [d for d in all_dataset_scores if d.endswith("_1") or d.endswith("U1")]
        val_datasets = [d for d in all_dataset_scores if d not in train_dataset]

        val_score_means = pd.DataFrame({d: all_dataset_scores[d] for d in val_datasets}).mean(axis=1).to_numpy()
        val_makespan_means = pd.DataFrame({d: all_dataset_makespans[d] for d in val_datasets}).mean(axis=1).to_numpy()

        fig, axs = plt.subplots(1, 2, figsize=(20, 12))
        axs[0].plot(existing_checkpoints, all_dataset_scores[train_dataset[0]], color='gray', linewidth=1, alpha=0.4, label='Training')
        axs[0].plot(existing_checkpoints, val_score_means, 'k--x', markersize=1, linewidth=1, label='Validation')
        axs[0].axvline(best_checkpoint, color='red', linestyle=':', linewidth=2)
        axs[0].set_xlabel("Episodes", fontsize=24)
        axs[0].set_ylabel("Cumulative Reward", fontsize=24)
        axs[0].legend(fontsize=24, frameon=True, loc='lower right')
        axs[0].tick_params(axis='both', labelsize=12)
        axs[0].set_ylim([
            min(min(all_dataset_scores[train_dataset[0]]), min(val_score_means)) - 2,
            max(max(all_dataset_scores[train_dataset[0]]), max(val_score_means)) + 5
        ])

        axs[1].plot(existing_checkpoints, all_dataset_makespans[train_dataset[0]], color='gray', linewidth=1, alpha=0.4, label='Training')
        axs[1].plot(existing_checkpoints, val_makespan_means, 'k--x', markersize=1, linewidth=1, label='Validation')
        axs[1].axvline(best_checkpoint, color='red', linestyle=':', linewidth=2)
        axs[1].set_xlabel("Episodes", fontsize=24)
        axs[1].set_ylabel("Makespan", fontsize=24)
        axs[1].legend(fontsize=24, frameon=True, loc='upper right')
        axs[1].tick_params(axis='both', labelsize=12)
        axs[1].set_ylim([
            min(min(all_dataset_makespans[train_dataset[0]]), min(val_makespan_means)) - 2,
            max(max(all_dataset_makespans[train_dataset[0]]), max(val_makespan_means)) + 5
        ])
        plt.tight_layout()

        result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{best_checkpoint}_{score_mean}_{score_std}_agent2_total_evaluation_plot.png"
        plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(plot_save_directory, exist_ok=True)
        file_path = os.path.join(plot_save_directory, result_name)
        plt.savefig(file_path)
        # plt.show()
        print(f"[Evaluate Agent2] best checkpoint: {best_checkpoint}")
        print(f"[Evaluate Agent2] best_score (mean): {score_mean}")
        print(f"[Evaluate Agent2] best_score (std): {score_std}")
        print(f"[Evaluate Agent2] best_makespan: {avg_makespans[best_idx]}")
        print("Agent 2 평가가 종료되었습니다.")
        best_path = os.path.join(save_directory_agent2, f"{best_checkpoint}family_param.pt")
        torch.save({'best_path': best_path}, os.path.join(save_directory_agent2, f"{best_checkpoint}_total_best_family_info.pt"))
        with open(os.path.join(plot_save_directory, "total_best_result_summary.txt"), "w") as f:
            f.write(f"Best checkpoint: {best_checkpoint}\n")
            f.write(f"Score mean: {score_mean}, std: {score_std}\n")
            f.write(f"Makespan: {avg_makespans[best_idx]}\n")

    @classmethod
    def get_result(cls, save_directory_agent1, save_directory_agent2, datasets, param):
        cls.init()
        # # === best_pm_info.pt 자동 탐색 ===
        # best_pm_info_file = next((f for f in os.listdir(save_directory_agent1) if f.endswith("_best_pm_info.pt")), None)
        # if best_pm_info_file is None:
        #     raise FileNotFoundError(f"No _best_pm_info.pt file found in {save_directory_agent1}")
        # best_pm_info_path = os.path.join(save_directory_agent1, best_pm_info_file)
        # best_pm_info = torch.load(best_pm_info_path)
        # best_pm_path = best_pm_info['best_path']
        # best_checkpoint_pm = best_pm_info_file.split('_')[0]

        pm_files = [f for f in os.listdir(save_directory_agent1) if f.endswith("pm_param.pt")]
        pm_files = sorted(pm_files, key=lambda x: int(x.replace("pm_param.pt", "")))
        last_pm_file = pm_files[-1]
        best_pm_path = os.path.join(save_directory_agent1, last_pm_file)

        # # === best_family_info.pt 자동 탐색 ===
        # best_family_info_file = next((f for f in os.listdir(save_directory_agent2) if f.endswith("_best_family_info.pt")), None)
        # if best_family_info_file is None:
        #     raise FileNotFoundError(f"No _best_family_info.pt file found in {save_directory_agent2}")
        #
        # best_family_info_path = os.path.join(save_directory_agent2, best_family_info_file)
        # best_family_info = torch.load(best_family_info_path)
        # best_family_path = best_family_info['best_path']
        # best_checkpoint_family = best_family_info_file.split('_')[0]

        # === param으로 받은 숫자를 기반으로 family checkpoint 직접 경로 구성 ===
        family_checkpoint_filename = f"{param}family_param.pt"
        best_family_path = os.path.join(save_directory_agent2, family_checkpoint_filename)
        if not os.path.exists(best_family_path):
            raise FileNotFoundError(f"{family_checkpoint_filename} not found in {save_directory_agent2}")
        best_checkpoint_family = str(param)

        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_pm_path, map_location=device))
        q_pm.eval()

        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        q_family.load_state_dict(torch.load(best_family_path, map_location=device))
        q_family.eval()

        print(f"[Result] agent1: {len(pm_files)}, agent2: {best_checkpoint_family} pt - {datasets} 데이터셋 평가 시작")
        for data_id in datasets:
            s = cls.env.reset(data_id)
            done = False
            score = 0.0
            pm_count = 0
            pm_end = False
            while not done:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s).float().to(device)
                    if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_end = True
                    a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                if a_pm == 0:
                    pm_count += 1
                    s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                    s = s_pm
                    score += r_pm
                else:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a_list = q_family.sample_action_list(s_tensor)
                    s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, 0, 'second')
                    s = s_prime
                    score += r_job
                if done:
                    break

            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            # 파일 경로에서 마지막 [] 안의 텍스트 추출
            match = re.search(r"\[([^\[\]]+)\]", save_directory_agent1.replace("\\", "/"))
            dataset_label = match.group(1) if match else "unknown"
            cls.env.gantt_chart(score, f'training_dataset_{dataset_label}')
            print(f"[Result] Dataset: {data_id}")
            print(f"[Result] Score: {score}")
            print(f"[Result] Makespan: {round(makespan, 3)}")

    @classmethod
    def script_performance(cls, env, n_epi, epsilon, memory, score, makespan_list, tardiness_list, score_list, pm_count):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        if isinstance(memory, MCReplayBuffer):
            buffer_size = len(memory.get_all_episodes())
        else:
            buffer_size = memory.size()
        output_string = f"n_episode: {n_epi}, score: {score:.3f}, n_buffer: {buffer_size}, eps: {epsilon * 100:.1f}%, makespan: {makespan:.3f}, pm count: {pm_count}"
        makespan_list[n_epi] = makespan
        tardiness_list[n_epi] = Total_Tardiness_time
        score_list[n_epi] = score
        if Parameters.log_on:
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
        elif agent_name == 'family' or agent_name == "fix_agent1_family":
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

        def plot_nan_filtered(ax, y, **kwargs):
            y = np.array(y)
            x = np.arange(len(y))
            ax.plot(x, y, **kwargs)  # keep NaN so line breaks automatically

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(68, 42))
        plot_nan_filtered(axs[0, 0], train_data, linestyle='--', marker='.', color='green')
        axs[0, 0].set_xlabel('Episodes', fontsize=36)
        axs[0, 0].set_ylabel(f'Training {object_title}', fontsize=36)
        axs[0, 0].tick_params(axis='both', labelsize=28)

        plot_nan_filtered(axs[1, 0], score_list, linestyle='-', marker='.', color='blue')
        axs[1, 0].set_xlabel('Episodes', fontsize=36)
        axs[1, 0].set_ylabel('Training Cumulative Reward', fontsize=36)
        axs[1, 0].tick_params(axis='both', labelsize=28)

        plot_nan_filtered(axs[0, 1], last_train_50, linestyle='--', marker='.', color='green')
        axs[0, 1].set_xlabel('Episodes', fontsize=36)
        axs[0, 1].set_ylabel(f'Average {object_title} in last 50 episodes', fontsize=36)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        plot_nan_filtered(axs[1, 1], last_50_score, linestyle='--', marker='.', color='blue')
        axs[1, 1].set_xlabel('Episodes', fontsize=36)
        axs[1, 1].set_ylabel('Average Cumulative Reward in last 50 episodes', fontsize=36)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red')
        axs[2, 0].set_xlabel('Episodes', fontsize=36)
        axs[2, 0].set_ylabel('Training Loss', fontsize=36)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        train_data = np.array(train_data)
        score_list = np.array(score_list)
        valid_corr = (~np.isnan(train_data)) & (~np.isnan(score_list))
        if np.any(valid_corr):
            correlation = np.corrcoef(train_data[valid_corr], score_list[valid_corr])[0, 1]
        else:
            correlation = 0

        axs[2, 1].scatter(train_data[valid_corr], score_list[valid_corr], color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        axs[2, 1].set_xlabel(f'{object_title}', fontsize=36)
        axs[2, 1].set_ylabel('Cumulative Reward', fontsize=36)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        axs[2, 1].legend(fontsize=36)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_{agent_name}_result.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path)
            print(f"{agent_name} 학습 그래프 저장 완료: {file_path}")

        # (1) Separate plot for axs[0, 1] (Average {object_title} in last 50 episodes)
        fig1, ax1 = plt.subplots(figsize=(20, 12))
        plot_nan_filtered(ax1, last_train_50, linestyle='--', marker='.', color='black')
        ax1.set_xlabel('The cumulative number of episodes', fontsize=36)
        ax1.set_ylabel(f'Average {object_title} in last 50 episodes', fontsize=36)
        ax1.tick_params(axis='both', labelsize=28)
        plt.tight_layout()
        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_{agent_name}_last50_{object_title.replace(' ', '_')}.png"
            file_path1 = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path1)
            print(f"{agent_name} last 50 {object_title} 그래프 저장 완료: {file_path1}")

        # (2) Separate plot for axs[1, 1] (Average Cumulative Reward in last 50 episodes)
        fig2, ax2 = plt.subplots(figsize=(20, 12))
        plot_nan_filtered(ax2, last_50_score, linestyle='--', marker='.', color='black')
        ax2.set_xlabel('The cumulative number of episodes', fontsize=36)
        ax2.set_ylabel('Average Cumulative Reward in last 50 episodes', fontsize=36)
        ax2.tick_params(axis='both', labelsize=28)
        plt.tight_layout()
        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_{agent_name}_last50_CumulativeReward.png"
            file_path2 = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path2)
            print(f"{agent_name} last 50 Reward 그래프 저장 완료: {file_path2}")

        # (3) Separate plot for axs[2, 1] (Scatter plot of Cumulative Reward vs {object_title})
        fig3, ax3 = plt.subplots(figsize=(20, 12))
        ax3.scatter(train_data[valid_corr], score_list[valid_corr], color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        ax3.set_xlabel(f'{object_title}', fontsize=36)
        ax3.set_ylabel('Cumulative Reward', fontsize=36)
        ax3.tick_params(axis='both', labelsize=28)
        ax3.legend(fontsize=32)
        plt.tight_layout()
        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_{agent_name}_CorrelationScatter.png"
            file_path3 = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path3)
            print(f"{agent_name} Correlation Scatter 그래프 저장 완료: {file_path3}")

        csv_data = pd.DataFrame({
            "episode": np.arange(len(train_data)),
            "train_data": train_data,  # makespan or total tardiness
            "score_list": score_list,  # cumulative reward
            "last_train_50": last_train_50,  # avg makespan / tardiness (last 50)
            "last_50_score": last_50_score  # avg reward (last 50)
        })
        if Parameters.save_plot:
            csv_name = (
                f"{Parameters.simulation_time}_"
                f"{Hyperparameters.object}_"
                f"{Hyperparameters.action_type}_"
                f"{Parameters.datasetId}_"
                f"A{Hyperparameters.rule_count}_"
                f"F{Hyperparameters.family_type}_"
                f"{agent_name}_running_data.csv"
            )
            csv_path = os.path.join(cls.running_data_save_directory, csv_name)
            csv_data.to_csv(csv_path, index=False)
            print(f"{agent_name} 학습 데이터 CSV 저장 완료: {csv_path}")

        if Parameters.gantt_on_check and agent_name != "pm":
            cls.env.gantt_chart(score_list[-1], agent_name)
        # plt.show()

    @classmethod
    def plot_rule_ratio_heatmap(cls, history, plot_save_directory):
        rules = sorted({k for d in history for k in d.keys()})
        M = np.array([[d.get(r, 0.0) for d in history] for r in rules], dtype=float)
        col_sums = M.sum(axis=0, keepdims=True)
        M = np.divide(M, np.where(col_sums == 0, 1, col_sums))

        fig, ax = plt.subplots(figsize=(6, 3.2))
        im = ax.imshow(M, aspect='auto', interpolation='nearest', vmin=0, vmax=1, cmap='YlGnBu')
        ax.set_xlabel('Episode', fontsize=10)

        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(rules, fontsize=10)
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=8)
        plt.tight_layout()
        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_F{Hyperparameters.family_type}_agent2_rule_ratio.png"
            file_path2 = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path2)
        plt.close()

    @classmethod
    def plot_rule_ratio_barplot(cls, ratio, plot_save_directory):
        # 원하는 rule 순서
        desired_order = ["SPTSDR", "SPTLDR", "LPTSDR", "LPTLDR"]

        # 순서대로 데이터 뽑기 (없으면 0으로 처리)
        rules = [r for r in desired_order if r in ratio]
        values = [ratio.get(r, 0.0) for r in rules]

        # 색상 매핑
        color_map = {
            "SPTSDR": "gray",
            "SPTLDR": "orange",
            "LPTSDR": "pink",
            "LPTLDR": "cyan"
        }
        colors = [color_map.get(rule, "lightgray") for rule in rules]

        fig, ax = plt.subplots(figsize=(8, 10))
        bars = ax.bar(rules, values, edgecolor='black', color=colors)

        ax.set_ylabel("Ratio", fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(rules)))
        ax.set_xticklabels(rules, fontsize=12, ha='right')
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()

        # legend에 수치 넣기
        legend_labels = [f"{rule}: {round(val, 2)}" for rule, val in zip(rules, values)]
        ax.legend(bars, legend_labels, fontsize=10, loc="upper right")

        if Parameters.save_plot:
            result_name = (
                f"{Parameters.simulation_time}_{Hyperparameters.object}_"
                f"{Hyperparameters.action_type}_{Parameters.datasetId}_"
                f"A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_"
                f"agent2_rule_ratio_bar.png"
            )
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path)
        plt.close()

    @classmethod
    def only_train_agent1(cls):
        cls.init()
        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}_agent1"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)

        score = 0.0
        pm_count = 0
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / (Hyperparameters.episode - 1))
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                pm_count = 0
                pm_end = False
                while not done:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_end = True
                    a = cls.q_pm.sample_action(s_tensor, epsilon, pm_end)
                    if a == 0:
                        pm_count += 1
                    s_prime, r, done, _, _ = cls.env.two_step_select(a, epsilon, 'first')
                    cls.memory_pm.put((s, a, r, s_prime, done))
                    s = s_prime
                    score += r
                    if done:
                        if pm_end == True:
                            print(f"Training Agent1 - Episode {n_epi} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break
                cls.memory_pm.end_episode()

            cls.first_mc_train(cls.q_pm, cls.memory_pm, cls.optimizer_pm, cls.loss_list_pm)
            cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm = cls.script_performance(cls.env, n_epi, epsilon, cls.memory_pm, score, cls.makespan_list_pm, cls.tardiness_list_pm, cls.score_list_pm, pm_count)
            cls.last_50_score_pm[n_epi] = (
                np.nanmean(cls.score_list_pm[max(0, n_epi - 49):n_epi + 1])
                if np.any(~np.isnan(cls.score_list_pm[max(0, n_epi - 49):n_epi + 1])) else np.nan
            )
            cls.last_50_makespan_pm[n_epi] = (
                np.nanmean(cls.makespan_list_pm[max(0, n_epi - 49):n_epi + 1])
                if np.any(~np.isnan(cls.makespan_list_pm[max(0, n_epi - 49):n_epi + 1])) else np.nan
            )
            cls.last_50_tardiness_pm[n_epi] = (
                np.nanmean(cls.tardiness_list_pm[max(0, n_epi - 49):n_epi + 1])
                if np.any(~np.isnan(cls.tardiness_list_pm[max(0, n_epi - 49):n_epi + 1])) else np.nan
            )
            if Parameters.param_down_on:
                params = cls.q_pm.state_dict()
                file_name = str(n_epi) + "pm_param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(params, file_path)

        cls.env.gantt_chart(score, f"result_agent1")
        print(f"[Agent1 Result] Dataset: {Parameters.datasetId}")
        print(f"[Agent1 Result] Score: {score}")
        print(f"[Agent1 Result] Makespan: {round(cls.makespan_list_pm[-1], 3)}")
        return f"{Parameters.simulation_time}_{Parameters.datasetId}"

    @classmethod
    def fix_agent1_main(cls, save_directory):
        cls.init()

        cls.save_directory_agent2 = os.path.join(pathConfig.model_save_path,f"{Parameters.simulation_time}_{Parameters.datasetId}_agent2")
        if Parameters.param_down_on:
            os.makedirs(cls.save_directory_agent2, exist_ok=True)

        cls.plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(cls.plot_save_directory, exist_ok=True)

        cls.fix_agent1_training_agent2(save_directory)
        cls.plot_results(cls.plot_save_directory, cls.loss_list_job, 'fix_agent1_family')

        best_checkpoint_family = cls.fix_agent1_evaluate_agent2(save_directory)
        makespan = cls.fix_agent1_result_agent2(save_directory, best_checkpoint_family, cls.plot_save_directory)
        return makespan

    @classmethod
    def fix_agent1_training_agent2(cls, save_directory):
        file_list = [f for f in os.listdir(save_directory) if os.path.isfile(os.path.join(save_directory, f))]
        file_list = [f for f in file_list if 'pm_param.pt' in f]
        file_list = sorted(file_list, key=lambda x: int(x.replace("pm_param.pt", "")))
        pm_check_point = len(file_list) - 1

        best_info_path = os.path.join(save_directory, f"{pm_check_point}pm_param.pt")
        print(f"[Agent2] Best Agent1 model loaded from: {best_info_path}")
        cls.q_pm.load_state_dict(torch.load(best_info_path, map_location=device))
        cls.q_pm.to(device)
        cls.q_pm.eval()

        score = 0.0
        pm_count = 0
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0, 1 - n_epi / (Hyperparameters.episode - 1))
            for dataid in Parameters.datasetId:
                s = cls.env.reset(dataid)
                done = False
                score = 0.0
                pm_count = 0
                pm_end = False
                while not done:
                    with torch.no_grad():  # Agent 1은 고정된 상태
                        s_tensor = torch.from_numpy(s).float().to(device)
                        if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                            pm_end = True
                        a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, epsilon, 'first')
                        s = s_pm
                        score += r_pm
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = cls.q_job.sample_action_list(s_tensor)
                        s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, epsilon, 'second')
                        done_mask = 1.0 if done else 0.0
                        cls.memory_job.put((s, a, r_job, s_prime, done_mask))
                        s = s_prime
                        score += r_job
                    if done:
                        if pm_end == True:
                            print(f"Training Agent2 - Episode {n_epi} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break

            if cls.memory_job.size() > Hyperparameters.buffer_limit / 10:
                cls.second_train(cls.q_job, cls.q_job_target, cls.memory_job, cls.optimizer_job, cls.loss_list_job)
                cls.makespan_list_job, cls.tardiness_list_job, cls.score_list_job = cls.script_performance(cls.env,n_epi,epsilon, cls.memory_job,score,cls.makespan_list_job,cls.tardiness_list_job,cls.score_list_job, pm_count)
                cls.last_50_score_job[n_epi] = (
                    np.nanmean(cls.score_list_job[max(0, n_epi - 49):n_epi + 1])
                    if np.any(~np.isnan(cls.score_list_job[max(0, n_epi - 49):n_epi + 1])) else np.nan
                )
                cls.last_50_makespan_job[n_epi] = (
                    np.nanmean(cls.makespan_list_job[max(0, n_epi - 49):n_epi + 1])
                    if np.any(~np.isnan(cls.makespan_list_job[max(0, n_epi - 49):n_epi + 1])) else np.nan
                )
                cls.last_50_tardiness_job[n_epi] = (
                    np.nanmean(cls.tardiness_list_job[max(0, n_epi - 49):n_epi + 1])
                    if np.any(~np.isnan(cls.tardiness_list_job[max(0, n_epi - 49):n_epi + 1])) else np.nan
                )
                if Parameters.param_down_on:
                    params = cls.q_job.state_dict()
                    file_name = str(n_epi) + "family_param.pt"
                    file_path = os.path.join(cls.save_directory_agent2, file_name)
                    torch.save(params, file_path)
            if n_epi % 20 == 0:
                cls.q_job_target.load_state_dict(cls.q_job.state_dict())

    @classmethod
    def fix_agent1_evaluate_agent2(cls, save_directory_agent1):
        file_list = [f for f in os.listdir(cls.save_directory_agent2) if os.path.isfile(os.path.join(cls.save_directory_agent2, f))]
        file_list = [f for f in file_list if 'family_param.pt' in f]
        check_point_list = sorted([int(f.replace("family_param.pt", "")) for f in file_list])

        file_list = [f for f in os.listdir(save_directory_agent1) if os.path.isfile(os.path.join(save_directory_agent1, f))]
        file_list = [f for f in file_list if 'pm_param.pt' in f]
        file_list = sorted(file_list, key=lambda x: int(x.replace("pm_param.pt", "")))
        pm_check_point = len(file_list) - 1

        best_pm_info_path = os.path.join(save_directory_agent1, f"{pm_check_point}pm_param.pt")
        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_pm_info_path, map_location=device))
        q_pm.eval()

        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)

        best_checkpoint = None
        best_score = float('-inf')
        best_makespan = float('inf')
        best_pm_count = 0

        evaluation_scores = {}  # checkpoint: score
        for check_point_number in check_point_list:
            check_point_path = os.path.join(cls.save_directory_agent2, f"{check_point_number}family_param.pt")
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
            total_score = 0.0
            for dataset in Parameters.datasetId:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0
                pm_end = False
                while not done:
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(s).float().to(device)
                        if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                            pm_end = True
                        a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                    if a_pm == 0:
                        pm_count += 1
                        s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                        s = s_pm
                        score += r_pm
                    else:
                        s_tensor = torch.from_numpy(s).float().to(device)
                        a_list = q_family.sample_action_list(s_tensor)
                        s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, 0, 'second')
                        s = s_prime
                        score += r_job
                    if done:
                        if pm_end == True:
                            print(f"Training Agent2 - Episode {check_point_number} PM 초과 여부: {pm_end}, count: {pm_count}")
                        break

                total_score += score
                total_score = np.round(total_score, decimals=3)

            evaluation_scores[check_point_number] = total_score
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

        # 전체 checkpoint 기준으로 x축 맞추기
        all_checkpoints = list(range(Hyperparameters.episode))
        y_scores = [evaluation_scores.get(cp, np.nan) for cp in all_checkpoints]

        if evaluation_scores:
            plt.figure(figsize=(24, 12))
            plt.plot(all_checkpoints, y_scores, marker='.', linestyle='--', color='blue')
            plt.xlabel("Episodes", fontsize=36)
            plt.ylabel("Cumulative Reward", fontsize=36)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.tight_layout()
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_agent2_evaluation_scores.png"
            file_path = os.path.join(cls.plot_save_directory, result_name)
            plt.savefig(os.path.join(file_path))
            # plt.show()

        print(f'[Evaluate Agent2] best checkpoint: {best_checkpoint}')
        print(f'[Evaluate Agent2] best_score: {best_score}')
        print(f'[Evaluate Agent2] best_makespan: {best_makespan}')
        print(f'[Evaluate Agent2] best_pm_count: {best_pm_count}')
        print("Agent 2 평가가 종료되었습니다.")

        best_path = os.path.join(cls.save_directory_agent2, f"{best_checkpoint}family_param.pt")
        torch.save({'best_path': best_path}, os.path.join(cls.save_directory_agent2, f"{best_checkpoint}_best_family_info.pt"))
        return best_checkpoint

    @classmethod
    def fix_agent1_result_agent2(cls, save_directory_agent1, best_checkpoint_family, plot_save_directory):
        file_list = [f for f in os.listdir(save_directory_agent1) if os.path.isfile(os.path.join(save_directory_agent1, f))]
        file_list = [f for f in file_list if 'pm_param.pt' in f]
        file_list = sorted(file_list, key=lambda x: int(x.replace("pm_param.pt", "")))
        pm_check_point = len(file_list) - 1

        best_info_path = os.path.join(save_directory_agent1, f"{pm_check_point}pm_param.pt")
        q_pm = Qnet(Hyperparameters.input_layer, 2).to(device)
        q_pm.load_state_dict(torch.load(best_info_path, map_location=device))
        q_pm.eval()

        # Family 네트워크 로딩
        best_family_info_path = os.path.join(cls.save_directory_agent2, f"{best_checkpoint_family}_best_family_info.pt")
        best_family_info = torch.load(best_family_info_path)
        best_family_path = best_family_info['best_path']
        q_family = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer).to(device)
        q_family.load_state_dict(torch.load(best_family_path, map_location=device))
        q_family.eval()

        print(f"[Result] agent1: {pm_check_point}, agent2: {best_checkpoint_family} pt 최종 평가 시작")
        for data_id in Parameters.datasetId:
            s = cls.env.reset(data_id)
            done = False
            score = 0.0
            pm_count = 0
            pm_end = False
            while not done:
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s).float().to(device)
                    if pm_count >= np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_end = True
                    a_pm = cls.q_pm.sample_action(s_tensor, 0, pm_end)
                if a_pm == 0:
                    pm_count += 1
                    s_pm, r_pm, done, _, _ = cls.env.two_step_select(a_pm, 0, 'first')
                    s = s_pm
                    score += r_pm
                else:
                    s_tensor = torch.from_numpy(s).float().to(device)
                    a_list = q_family.sample_action_list(s_tensor)
                    s_prime, r_job, done, a, _ = cls.env.two_step_select(a_list, 0, 'second')
                    s = s_prime
                    score += r_job

            Flow_time, util, makespan, tardiness, lateness, t_max = cls.env.performance_measure()
            cls.env.gantt_chart(score, f'fix_agent1_family_{best_checkpoint_family}result_agent2')
            episode_ratio = cls.env.get_rule_ratio_dist()
            cls.plot_rule_ratio_barplot(episode_ratio, cls.plot_save_directory)
            print(f"[Result] Dataset: {data_id}")
            print(f"[Result] Score: {score}")
            print(f"[Result] Makespan: {round(makespan, 3)}")
            print(f"[Result] PM count: {pm_count}")
            print(f"[Result] Rule ratio: {episode_ratio}")
            return round(makespan, 3)