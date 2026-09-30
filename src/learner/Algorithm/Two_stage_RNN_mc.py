import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from src.simlator.Simulator import *
from src.common.Parameters import *
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RNN_ActorCritic(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, fc_size=64):
        super(RNN_ActorCritic, self).__init__()
        self.actor_gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1)
        )

        self.critic_gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1)
        )

    def forward_actor(self, state, lengths, mask):
        state = state.to(device)
        mask = mask.to(device)
        packed = nn.utils.rnn.pack_padded_sequence(state, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.actor_gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        priorities = self.actor_fc(out).squeeze(-1)
        probs = self.masked_softmax(priorities, mask)
        return probs

    def forward_critic(self, state, lengths):
        state = state.to(device)
        packed = nn.utils.rnn.pack_padded_sequence(state, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.critic_gru(packed)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        value = self.critic_fc(hidden_cat).squeeze(-1)
        return value

    def masked_softmax(self, priorities, mask):
        mask = mask.to(dtype=torch.bool) if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool, device=device)
        action_probs = priorities.masked_fill(mask == 0, float('-inf'))
        probs = torch.nn.functional.softmax(action_probs, dim=-1)
        return probs

    def act(self, state, lengths, available_jobs_mask):
        state = state.to(device)
        if not isinstance(available_jobs_mask, torch.Tensor):
            available_jobs_mask = torch.tensor(available_jobs_mask, dtype=torch.float32, device=device)
        else:
            available_jobs_mask = available_jobs_mask.clone().detach().to(dtype=torch.float32, device=device)
        priorities = self.forward_actor(state, lengths, available_jobs_mask)
        probs = self.masked_softmax(priorities, available_jobs_mask)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs

class Two_Stage_RNN_MC:
    hidden_size = 32
    fc_size = 64
    learning_rate = 0.0001
    gamma = 0.99
    clip_epsilon = 0.3
    num_first_episodes = 500
    num_second_episodes = 2000
    update_steps = 5
    batch_size = 5
    entropy_coef = 0.01

    memory_first_stage = []
    memory_second_stage = []

    @classmethod
    def pad_states(cls, state_list):
        lengths = [s.shape[0] for s in state_list]
        max_len = max(lengths)
        batch_states = []
        for s in state_list:
            if s.shape[0] < max_len:
                pad = np.zeros((max_len - s.shape[0], s.shape[1]), dtype=s.dtype)
                s_padded = np.concatenate([s, pad], axis=0)
            else:
                s_padded = s
            batch_states.append(s_padded)
        batch_states = np.stack(batch_states)  # (batch, max_len, input_size)
        return torch.tensor(batch_states, dtype=torch.float32).to(device), lengths

    @classmethod
    def make_batch(cls):
        s_lst, a_lst, r_lst, log_a_lst, done_lst, mask_lst = [], [], [], [], [], []
        for transition in cls.memory:
            s, a, r, log_a, done_mask, mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            log_a_lst.append([log_a])
            done_lst.append([done_mask])
            mask_lst.append(mask)

        states, lengths = cls.pad_states(s_lst)
        actions = torch.tensor(np.array(a_lst), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(r_lst), dtype=torch.float32).to(device)
        log_probs_old = torch.tensor(np.array(log_a_lst), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(done_lst), dtype=torch.float32).to(device)
        masks = torch.stack([torch.tensor(m, dtype=torch.float32) for m in mask_lst]).to(device)

        num_samples = states.size(0)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        batch_indices = [indices[i:i + cls.batch_size] for i in range(0, num_samples, cls.batch_size)]
        return states, lengths, actions, rewards, log_probs_old, dones, masks, batch_indices

    @classmethod
    def train_stage(cls, model, optimizer, memory, loss_list):
        cls.memory = memory
        states, lengths, actions, rewards, log_probs_old, dones, masks, _ = cls.make_batch()

        returns = []
        G = 0
        for reward, d in zip(reversed(rewards.squeeze()), reversed(dones.squeeze())):
            G = reward + cls.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        with torch.no_grad():
            state_values = cls.model.forward_critic(states, lengths)
        advantages = returns - state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(cls.update_steps):
            _, _, action_probs = cls.model.act(states, lengths, masks)
            log_action_probs = torch.log(action_probs + 1e-8)
            log_pi = log_action_probs.gather(1, actions).squeeze(1)

            ratio = torch.exp(log_pi - log_probs_old.squeeze(1))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cls.clip_epsilon, 1 + cls.clip_epsilon) * advantages
            entropy = -(action_probs * log_action_probs).sum(dim=1).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - cls.entropy_coef * entropy

            values_pred = cls.model.forward_critic(states, lengths)
            critic_loss = torch.nn.functional.smooth_l1_loss(values_pred, returns)

            loss = actor_loss + critic_loss
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        cls.memory.clear()

    @classmethod
    def main(cls):
        train_test_start_time = time.time()
        cls.env = Simulator()
        cls.model = RNN_ActorCritic(input_size=6, hidden_size=cls.hidden_size, fc_size=cls.fc_size).to(device)
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=cls.learning_rate)

        first_train_makespan_list, first_train_score_list, first_train_tardiness_list, first_train_loss_list = [], [], [], []
        first_train_last_50_makespan, first_train_last_50_score, first_train_last_50_tardiness = [], [], []

        second_train_makespan_list, second_train_score_list, second_train_tardiness_list, second_train_loss_list = [], [], [], []
        second_train_last_50_makespan, second_train_last_50_score, second_train_last_50_tardiness = [], [], []

        first_test_score_list, first_test_makespan_list, first_test_tardiness_list = [], [], []
        first_test_last_50_score, first_test_last_50_makespan, first_test_last_50_tardiness = [], [], []

        second_test_score_list, second_test_makespan_list, second_test_tardiness_list = [], [], []
        second_test_last_50_score, second_test_last_50_makespan, second_test_last_50_tardiness = [], [], []

        learn_iter = 0

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)

        cls.plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(cls.plot_save_directory, exist_ok=True)

        first_score, second_score = 0.0, 0.0
        first_pm_count = 0
        for n_epi in range(cls.num_first_episodes):
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                first_score = 0.0
                first_pm_count = 0
                pm_count = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    if first_pm_count > np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_count = True
                    available_job = cls.env.available_jobs_mask(pm_count)
                    action, log_prob, _ = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    if action == len(available_job) - 1:
                        first_pm_count += 1
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action.item(), n_epi, pm_count)
                    done_mask = 1.0 if done else 0.0
                    cls.memory_first_stage.append((state, action.item(), reward, log_prob.item(), done_mask, available_job))
                    state = next_state
                    first_score += reward
                    if done:
                        break

            cls.train_stage(cls.model, cls.optimizer, cls.memory_first_stage, first_train_loss_list)
            learn_iter += 1
            first_train_makespan_list, first_train_score_list, first_train_tardiness_list = cls.script_performance(cls.env, n_epi, cls.memory_first_stage, first_score, first_train_makespan_list, first_train_score_list, first_train_tardiness_list)
            if Parameters.log_on:
                print(f'Episode {n_epi}, First learn count: {learn_iter}, Score: {first_score:.3f}, Makespan: {first_train_makespan_list[-1]:.3f}, PM count: {first_pm_count}')
            first_train_last_50_makespan.append(np.mean(first_train_makespan_list[-50:]))
            first_train_last_50_score.append(np.mean(first_train_score_list[-50:]))
            first_train_last_50_tardiness.append(np.mean(first_train_tardiness_list[-50:]))

            if Parameters.param_down_on:
                file_name = str(n_epi) + "param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(cls.model.state_dict(), file_path)

            cls.model.eval()
            first_test_score = 0.0
            first_test_pm_count = 0
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                first_test_score = 0.0
                first_test_pm_count = 0
                test_pm_count = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    if first_test_pm_count > np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        test_pm_count = True
                    available_job = cls.env.available_jobs_mask(test_pm_count)
                    _, _, action_probs = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    action = torch.argmax(action_probs, dim=-1).item()
                    if action == len(available_job) - 1:
                        first_test_pm_count += 1
                        #print('first_test_pm_count: ', first_test_pm_count)
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action, n_epi, test_pm_count)
                    state = next_state
                    first_test_score += reward
                    if done:
                        break
            first_test_makespan_list, first_test_score_list, first_test_tardiness_list = cls.script_performance(cls.env, n_epi, cls.memory_first_stage, first_test_score, first_test_makespan_list, first_test_score_list, first_test_tardiness_list)
            if Parameters.log_on:
                print(f'Test Episode {n_epi}, First learn count: {learn_iter}, Test Score: {first_test_score:.3f}, Test Makespan: {first_test_makespan_list[-1]:.3f}, Test PM count: {first_test_pm_count}')
                print('=============================================================================')
            first_test_last_50_score.append(np.round(np.mean(first_test_score_list[-50:]), 3))
            first_test_last_50_makespan.append(np.round(np.mean(first_test_makespan_list[-50:]), 3))
            first_test_last_50_tardiness.append(np.round(np.mean(first_test_tardiness_list[-50:]), 3))
            cls.model.train()

        object_name = 'Total Tardiness' if Hyperparameters.object == 'total_tardiness' else 'Makespan'
        # First Training 시각화
        first_train_data = first_train_tardiness_list if object_name == 'Total Tardiness' else first_train_makespan_list
        first_last_train_20 = first_train_last_50_tardiness if object_name == 'Total Tardiness' else first_train_last_50_makespan
        cls.plot_training_progress(object_name, 'First', first_train_data, first_last_train_20, first_train_score_list, first_train_last_50_score, first_train_loss_list)
        # First Test 시각화
        first_test_data = first_test_tardiness_list if object_name == 'Total Tardiness' else first_test_makespan_list
        first_last_test_50 = first_test_last_50_tardiness if object_name == 'Total Tardiness' else first_test_last_50_makespan
        cls.plot_test_progress(object_name, 'First', first_test_data, first_last_test_50, first_test_score_list, first_test_last_50_score)
        if Parameters.gantt_on_check:
            cls.env.gantt_chart(first_test_score, 'First')

        learn_iter = 0
        second_pm_count = 0
        for n_epi in range(cls.num_first_episodes, cls.num_first_episodes + cls.num_second_episodes):
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                second_score = 0.0
                second_pm_count = 0
                pm_count = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    if second_pm_count > np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_count = True
                    available_job = cls.env.available_jobs_mask(pm_count)
                    action, log_prob, _ = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    if action == len(available_job) - 1:
                        second_pm_count += 1
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action.item(), n_epi, pm_count)
                    done_mask = 1.0 if done else 0.0
                    cls.memory_second_stage.append((state, action.item(), reward, log_prob.item(), done_mask, available_job))
                    state = next_state
                    second_score += reward
                    if done:
                        break

            cls.train_stage(cls.model, cls.optimizer, cls.memory_second_stage, second_train_loss_list)
            learn_iter += 1
            second_train_makespan_list, second_train_score_list, second_train_tardiness_list = cls.script_performance(cls.env, n_epi, cls.memory_second_stage, second_score, second_train_makespan_list, second_train_score_list, second_train_tardiness_list)
            if Parameters.log_on:
                print(f'Episode {n_epi}, Second learn count: {learn_iter}, Score: {second_score:.3f}, Makespan: {second_train_makespan_list[-1]:.3f}, PM count: {second_pm_count}')
            second_train_last_50_makespan.append(np.mean(second_train_makespan_list[-50:]))
            second_train_last_50_score.append(np.mean(second_train_score_list[-50:]))
            second_train_last_50_tardiness.append(np.mean(second_train_tardiness_list[-50:]))

            if Parameters.param_down_on:
                file_name = str(n_epi) + "param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(cls.model.state_dict(), file_path)

            cls.model.eval()
            second_test_score = 0.0
            second_test_pm_count = 0
            second_pm_count = False
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                second_test_score = 0.0
                second_test_pm_count = 0
                second_pm_count = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    if second_test_pm_count > np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        second_pm_count = True
                    available_job = cls.env.available_jobs_mask(second_pm_count)
                    _, _, action_probs = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    action = torch.argmax(action_probs, dim=-1).item()
                    if action == len(available_job) - 1:
                        second_test_pm_count += 1
                        # print('second_test_pm_count: ', second_test_pm_count)
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action, n_epi, second_pm_count)
                    state = next_state
                    second_test_score += reward
                    if done:
                        break
            second_test_makespan_list, second_test_score_list, second_test_tardiness_list  = cls.script_performance(cls.env, n_epi, cls.memory_second_stage, second_test_score, second_test_makespan_list, second_test_score_list, second_test_tardiness_list)
            if Parameters.log_on:
                print(f'Test Episode {n_epi}, Second learn count: {learn_iter}, Test Score: {second_test_score:.3f}, Test Makespan: {second_test_makespan_list[-1]:.3f}, Test PM count: {second_test_pm_count}')
                print('=============================================================================')
            second_test_last_50_score.append(np.round(np.mean(second_test_score_list[-50:]), 3))
            second_test_last_50_makespan.append(np.round(np.mean(second_test_makespan_list[-50:]), 3))
            second_test_last_50_tardiness.append(np.round(np.mean(second_test_tardiness_list[-50:]), 3))
            cls.model.train()

        object_name = 'Total Tardiness' if Hyperparameters.object == 'total_tardiness' else 'Makespan'
        # Second Training 시각화
        second_train_data = second_train_tardiness_list if object_name == 'Total Tardiness' else second_train_makespan_list
        second_last_train_50 = second_train_last_50_tardiness if object_name == 'Total Tardiness' else second_train_last_50_makespan
        cls.plot_training_progress(object_name, 'Second', second_train_data, second_last_train_50, second_train_score_list, second_train_last_50_score, second_train_loss_list)

        # Second Test 시각화
        second_test_data = second_test_tardiness_list if object_name == 'Total Tardiness' else second_test_makespan_list
        second_last_test_50 = second_test_last_50_tardiness if object_name == 'Total Tardiness' else second_test_last_50_makespan
        cls.plot_test_progress(object_name, 'Second', second_test_data, second_last_test_50, second_test_score_list, second_test_last_50_score)
        if Parameters.gantt_on_check:
            cls.env.gantt_chart(second_test_score, 'Second')
        train_test_end_time = time.time()
        train_elapsed_time = train_test_end_time - train_test_start_time
        print(f"model train time: {train_elapsed_time:.4f} sec")

        cls.running_data_save_directory = f"{pathConfig.running_data_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(cls.running_data_save_directory, exist_ok=True)

        csv_data = pd.DataFrame({
            "episode": np.arange(cls.num_first_episodes, cls.num_first_episodes+len(second_train_makespan_list)),

            # Train
            "train_makespan": second_train_makespan_list,
            "train_cum_reward": second_train_score_list,

            # Test
            "test_makespan": second_test_makespan_list,
            "test_cum_reward": second_test_score_list,
        })
        csv_name = (
            f"{Parameters.simulation_time}_"
            f"{Hyperparameters.object}_"
            f"{Hyperparameters.action_type}_"
            f"{Parameters.datasetId}_"
            f"F{Hyperparameters.family_type}_"
            f"rnn_second_stage_data.csv"
        )

        csv_path = os.path.join(cls.running_data_save_directory, csv_name)
        csv_data.to_csv(csv_path, index=False)

        print(f"rnn Second stage CSV 저장 완료: {csv_path}")

        # 테스트 결과 중 최고 성능 모델 인덱스 찾기
        best_index = int(np.nanargmax(second_test_score_list))  # 또는 np.nanargmin() if lower score is better
        best_score = second_test_score_list[best_index]
        best_param_filename = f"{cls.num_first_episodes + best_index}param.pt"
        best_param_path = os.path.join(save_directory, best_param_filename)
        print(f"\n✅ Best Second Test Score: {best_score:.3f} at Episode {cls.num_first_episodes + best_index}")
        print(f"🔍 Corresponding Model File: {best_param_path}")
        # 모델 로드
        # ============================================================
        # Best checkpoint 로드
        # ============================================================
        best_episode = cls.num_first_episodes + best_index

        cls.model.load_state_dict(
            torch.load(
                best_param_path,
                map_location=device,
                weights_only=True
            )
        )
        cls.model.eval()


        # ============================================================
        # Best model 별도 저장
        # ============================================================
        best_model_path = os.path.join(
            save_directory,
            "best_param.pt"
        )

        torch.save(
            cls.model.state_dict(),
            best_model_path
        )


        # ============================================================
        # Best episode 번호 저장
        # ============================================================
        best_episode_path = os.path.join(
            save_directory,
            "best_episode.txt"
        )

        with open(best_episode_path, "w", encoding="utf-8") as f:
            f.write(str(best_episode))


        print(f"[Best Model Saved] {best_model_path}")
        print(f"[Best Episode Saved] {best_episode}")
        # best 선택 ~ 결과 평가 시간 측정 시작
        best_eval_start_time = time.time()

        if Parameters.gantt_on_check:
            best_score_for_gantt = 0.0
            best_score_pm_count = 0
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                pm_count = False
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    if best_score_pm_count > np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job):
                        pm_count = True
                    available_job = cls.env.available_jobs_mask(pm_count)  # PM count는 리셋된 상태 기준
                    _, _, action_probs = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    action = torch.argmax(action_probs, dim=-1).item()
                    if action == len(available_job) - 1:
                        best_score_pm_count += 1
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action, cls.num_first_episodes + best_index, pm_count)
                    state = next_state
                    best_score_for_gantt += reward
            Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = cls.env.performance_measure()
            print(f"[Result] Dataset: {dataid}")
            print(f"[Result] Score: {best_score_for_gantt}")
            print(f"[Result] Makespan: {round(makespan, 3)}")
            # 실행 시간 측정 종료
            best_eval_end_time = time.time()
            best_eval_elapsed_time = best_eval_end_time - best_eval_start_time
            print(f"Best model evaluation time: {best_eval_elapsed_time:.4f} sec")
            cls.env.gantt_chart(best_score_for_gantt, None)

            txt_name = (
                f"{Parameters.simulation_time}_"
                f"{Hyperparameters.object}_"
                f"{Hyperparameters.action_type}_"
                f"{Parameters.datasetId}_"
                f"F{Hyperparameters.family_type}_"
                f"rnn_second_stage_best_result.txt"
            )
            txt_path = os.path.join(cls.running_data_save_directory, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("===== Second Stage Result =====\n")
                f.write(f"training Time (sec): {train_elapsed_time:.6f}\n")
                f.write(f"Evaluation Time (sec): {best_eval_elapsed_time:.6f}\n")
            print(f"📄 Best result TXT 저장 완료: {txt_path}")
        return makespan

    @classmethod
    def script_performance(cls, env, n_epi, memory, score, makespan_list, score_list, tardiness_list):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        if Hyperparameters.object == 'total_tardiness':
            output_string = f"n_episode: {n_epi}, n_buffer: {len(memory)}, score: {score:.3f}, total tardiness: {Total_Tardiness_time:.3f}\n" + \
                            f"--------------------------------------------------------------------"
        else:
            output_string = f"n_episode: {n_epi}, n_buffer: {len(memory)}, score: {score:.3f}, makespan: {makespan:.3f}\n" + \
                            f"--------------------------------------------------------------------"
        makespan_list.append(makespan)
        tardiness_list.append(Total_Tardiness_time)
        score_list.append(score)
        if Parameters.log_on:
            # print(output_string)
            logging.info(f'performance :{output_string}')
        return makespan_list, score_list, tardiness_list

    @classmethod
    def plot_training_progress(cls, object_name, phase, train_data, last_train_50, score_list, last_score_50, loss_list):
        fig, axs = plt.subplots(3, 2, figsize=(58, 26))

        axs[0, 0].plot(range(len(train_data)), train_data, linestyle='--', marker='.', color='green', label=f'Training {object_name}')
        axs[0, 0].set_xlabel('Episodes', fontsize=32)
        axs[0, 0].set_ylabel(f'{phase} Training {object_name}', fontsize=32)
        axs[0, 0].tick_params(axis='both', labelsize=28)

        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='blue', label='Training Score')
        axs[1, 0].set_xlabel('Episodes', fontsize=32)
        axs[1, 0].set_ylabel(f'{phase} Training Score', fontsize=32)
        axs[1, 0].tick_params(axis='both', labelsize=28)

        axs[0, 1].plot(range(len(last_train_50)), last_train_50, linestyle='--', marker='.', color='green', label=f'Training {object_name}')
        axs[0, 1].set_xlabel('Episodes', fontsize=32)
        axs[0, 1].set_ylabel(f'Average {object_name} over previous 50 episodes', fontsize=32)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        axs[1, 1].plot(range(len(last_score_50)), last_score_50, linestyle='--', marker='.', color='blue', label='Training Score')
        axs[1, 1].set_xlabel('Episodes', fontsize=32)
        axs[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=32)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red', label='Training Loss')
        axs[2, 0].set_xlabel('Episodes', fontsize=32)
        axs[2, 0].set_ylabel(f'{phase} Training Loss', fontsize=32)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        correlation = np.corrcoef(train_data, score_list)[0, 1]
        axs[2, 1].scatter(train_data, score_list, color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        axs[2, 1].set_xlabel(object_name, fontsize=32)
        axs[2, 1].set_ylabel('Score', fontsize=32)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        axs[2, 1].legend(fontsize=28)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name_1 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_{phase}_train_result.png"
            file_path_1 = os.path.join(cls.plot_save_directory, result_name_1)
            fig.savefig(file_path_1)  # Training plot 저장
            print(f"{phase} 그래프가 저장되었습니다: {file_path_1}")
        # plt.show()

    @classmethod
    def plot_test_progress(cls, object_name, phase, test_data, last_test_50, score_list, last_score_50):
        fig, axs = plt.subplots(2, 2, figsize=(48, 24))

        axs[0, 0].plot(range(len(test_data)), test_data, linestyle='--', marker='.', color='purple')
        axs[0, 0].set_xlabel('Episodes', fontsize=28)
        axs[0, 0].set_ylabel(f'{phase} Test {object_name}', fontsize=28)
        axs[0, 0].tick_params(axis='both', labelsize=24)

        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='orange')
        axs[1, 0].set_xlabel('Episodes', fontsize=28)
        axs[1, 0].set_ylabel(f'{phase} Test Score', fontsize=28)
        axs[1, 0].tick_params(axis='both', labelsize=24)

        axs[0, 1].plot(range(len(last_test_50)), last_test_50, linestyle='--', marker='.', color='purple')
        axs[0, 1].set_xlabel('Episodes', fontsize=28)
        axs[0, 1].set_ylabel(f'Avg {object_name} over last 50', fontsize=28)
        axs[0, 1].tick_params(axis='both', labelsize=24)

        axs[1, 1].plot(range(len(last_score_50)), last_score_50, linestyle='--', marker='.', color='orange')
        axs[1, 1].set_xlabel('Episodes', fontsize=28)
        axs[1, 1].set_ylabel('Avg Score over last 50', fontsize=28)
        axs[1, 1].tick_params(axis='both', labelsize=24)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_{phase}_test_result.png"
            file_path = os.path.join(cls.plot_save_directory, result_name)
            fig.savefig(file_path)
            print(f"{phase} 테스트 그래프 저장 완료: {file_path}")
        # plt.show()

    # @classmethod
    # def run_trained_policy(cls, new_dataset_ids, model_directory, param=None):
    #     """
    #     학습 완료된 RNN 정책을 새로운 데이터셋에 적용한다.
    #     재학습 없이 greedy inference만 수행한다.
    #     """
    #     cls.env = Simulator()

    #     if not isinstance(new_dataset_ids, (list, tuple)):
    #         new_dataset_ids = [new_dataset_ids]


    #     best_model_path = os.path.join(model_directory, "best_param.pt")
    #     best_episode_path = os.path.join(model_directory, "best_episode.txt")

    #     if not os.path.exists(best_model_path):
    #         raise FileNotFoundError(
    #             f"best_param.pt가 없습니다: {best_model_path}. "
    #             f"기존 학습 결과라면 param에 best episode 번호를 직접 넣어주세요."
    #         )

    #     if not os.path.exists(best_episode_path):
    #         raise FileNotFoundError(
    #             f"best_episode.txt가 없습니다: {best_episode_path}. "
    #             f"기존 학습 결과라면 param에 best episode 번호를 직접 넣어주세요."
    #         )

    #     with open(best_episode_path, "r", encoding="utf-8") as f:
    #         inference_episode = int(f.read().strip())

    #     model_path = best_model_path

    #     if not os.path.exists(model_path):
    #         raise FileNotFoundError(
    #             f"RNN checkpoint를 찾을 수 없습니다: {model_path}"
    #         )

    #     # 학습된 RNN 모델 로드
    #     model = RNN_ActorCritic(
    #         input_size=6,
    #         hidden_size=cls.hidden_size,
    #         fc_size=cls.fc_size
    #     ).to(device)

    #     model.load_state_dict(
    #         torch.load(
    #             model_path,
    #             map_location=device,
    #             weights_only=True
    #         )
    #     )
    #     model.eval()

    #     print("=" * 80)
    #     print("[RNN Trained Policy Inference]")
    #     print(f"Model   : {model_path}")
    #     print(f"Episode : {inference_episode}")
    #     print("=" * 80)

    #     results = {}

    #     for data_id in new_dataset_ids:
    #         state = cls.env.reset(data_id)

    #         done = False
    #         score = 0.0
    #         pm_count = 0
    #         pm_end = False

    #         while not done:
    #             state_tensor = torch.tensor(
    #                 state,
    #                 dtype=torch.float32
    #             ).unsqueeze(0).to(device)

    #             if (pm_count > np.sqrt(Simulator.number_of_machine) * np.sqrt(Simulator.number_of_job)):
    #                 pm_end = True

    #             available_job = cls.env.available_jobs_mask(pm_end)
    #             available_job_tensor = torch.tensor(
    #                 available_job,
    #                 dtype=torch.float32,
    #                 device=device
    #             )
    #             # greedy inference: 가장 높은 확률의 action 선택
    #             with torch.no_grad():
    #                 action_probs = model.forward_actor(state_tensor, [state.shape[0]], available_job_tensor)
    #                 action = torch.argmax(action_probs, dim=-1).item()

    #             if action == len(available_job) - 1:
    #                 pm_count += 1

    #             next_state, reward, done, _ = cls.env.two_stage_rnn(action, inference_episode, pm_end)
    #             state = next_state
    #             score += reward

    #         (
    #             total_flow_time,
    #             util,
    #             makespan,
    #             total_tardiness,
    #             total_lateness,
    #             t_max
    #         ) = cls.env.performance_measure()

    #         results[data_id] = {
    #             "score": score,
    #             "makespan": makespan,
    #             "tardiness": total_tardiness,
    #             "flow_time": total_flow_time,
    #             "util": util,
    #             "lateness": total_lateness,
    #             "t_max": t_max,
    #             "pm_count": pm_count,
    #             "checkpoint": inference_episode
    #         }

    #         print(f"[Inference Result] Dataset    : {data_id}")
    #         print(f"[Inference Result] Checkpoint : {inference_episode}")
    #         print(f"[Inference Result] Score      : {score:.3f}")
    #         print(f"[Inference Result] Makespan   : {makespan:.3f}")
    #         print(f"[Inference Result] Tardiness  : {total_tardiness:.3f}")
    #         print(f"[Inference Result] PM count   : {pm_count}")

    #         if Parameters.gantt_on_check:
    #             cls.env.gantt_chart(
    #                 score,
    #                 f"RNN_inference_{data_id}"
    #             )
    #     return makespan

    @classmethod
    def run_trained_policy(cls, new_dataset_ids, model_directory, param=None):
        """
        학습 종료 후 best result 평가와 동일한 방식으로 inference 수행.
        """

        cls.env = Simulator()

        if not isinstance(new_dataset_ids, (list, tuple)):
            new_dataset_ids = [new_dataset_ids]

        if param is None:
            best_model_path = os.path.join(model_directory, "best_param.pt")
            best_episode_path = os.path.join(model_directory, "best_episode.txt")

            if not os.path.exists(best_model_path):
                raise FileNotFoundError(
                    f"best_param.pt가 없습니다: {best_model_path}"
                )

            if not os.path.exists(best_episode_path):
                raise FileNotFoundError(
                    f"best_episode.txt가 없습니다: {best_episode_path}"
                )

            with open(best_episode_path, "r", encoding="utf-8") as f:
                inference_episode = int(f.read().strip())

            model_path = best_model_path
        else:
            inference_episode = int(param)
            model_path = os.path.join(
                model_directory,
                f"{inference_episode}param.pt"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RNN checkpoint를 찾을 수 없습니다: {model_path}"
            )

        # 학습 result와 동일하게 cls.model에 로드
        cls.model = RNN_ActorCritic(
            input_size=6,
            hidden_size=cls.hidden_size,
            fc_size=cls.fc_size
        ).to(device)

        cls.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
                weights_only=True
            )
        )
        cls.model.eval()

        print("=" * 80)
        print("[RNN Trained Policy Inference - Same as Training Result]")
        print(f"Model   : {model_path}")
        print(f"Episode : {inference_episode}")
        print("=" * 80)

        results = {}

        for data_id in new_dataset_ids:
            state = cls.env.reset(data_id)
            done = False

            # 학습 result와 동일한 변수 구조
            best_score_for_gantt = 0.0
            best_score_pm_count = 0
            pm_count = False

            while not done:
                state_tensor = torch.tensor(
                    state,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                if (
                    best_score_pm_count
                    > np.sqrt(Simulator.number_of_machine)
                    * np.sqrt(Simulator.number_of_job)
                ):
                    pm_count = True

                available_job = cls.env.available_jobs_mask(pm_count)

                # 학습 result와 동일:
                # act() 호출 후 action_probs에서 argmax 선택
                _, _, action_probs = cls.model.act(
                    state_tensor,
                    [state.shape[0]],
                    available_job
                )

                action = torch.argmax(
                    action_probs,
                    dim=-1
                ).item()

                if action == len(available_job) - 1:
                    best_score_pm_count += 1

                next_state, reward, done, _ = cls.env.two_stage_rnn(
                    action,
                    inference_episode,
                    pm_count
                )

                state = next_state
                best_score_for_gantt += reward

            (
                total_flow_time,
                util,
                makespan,
                total_tardiness,
                total_lateness,
                t_max
            ) = cls.env.performance_measure()

            results[data_id] = {
                "score": best_score_for_gantt,
                "makespan": makespan,
                "tardiness": total_tardiness,
                "flow_time": total_flow_time,
                "util": util,
                "lateness": total_lateness,
                "t_max": t_max,
                "pm_count": best_score_pm_count,
                "checkpoint": inference_episode
            }

            print(f"[Inference Result] Dataset    : {data_id}")
            print(f"[Inference Result] Checkpoint : {inference_episode}")
            print(f"[Inference Result] Score      : {best_score_for_gantt:.3f}")
            print(f"[Inference Result] Makespan   : {makespan:.3f}")
            print(f"[Inference Result] Tardiness  : {total_tardiness:.3f}")
            print(f"[Inference Result] PM count   : {best_score_pm_count}")

            if Parameters.gantt_on_check:
                cls.env.gantt_chart(
                    best_score_for_gantt,
                    f"RNN_inference"
                )

        return makespan