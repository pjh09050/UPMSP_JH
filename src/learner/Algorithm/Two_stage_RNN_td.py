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

class RNN_ActorCritic(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, fc_size=64):
        super(RNN_ActorCritic, self).__init__()
        self.actor_gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
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

    def forward_actor(self, state, lengths):
        # state: (batch, seq_len, input_size)
        packed = nn.utils.rnn.pack_padded_sequence(state, lengths, batch_first=True, enforce_sorted=False) # 패딩된 값들은 무시하고 유효환 값들만 처리 # enforce_sorted : lengths가 내림차순이 아닐수도 있으므로 False
        packed_out, _ = self.actor_gru(packed) # GRU가 PackedSequence를 받았으므로 패딩된 부분은 자동으로 무시하고 모든 스텝의 은닉 상태(hidden state) 연산 수행
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # PackedSequence를 원래 패딩된 형태 (batch, seq_len, hidden_size*2)로 변환 # 모든 타입 스텝의 출력을 사용해야하므로 다시 pad
        # 각 타임스텝에 대해 actor_fc 적용 → (batch, seq_len, 1)
        priorities = self.actor_fc(out).squeeze(-1)  # squeeze(-1)을 사용하여 마지막 차원 제거 → (batch, seq_len)
        return priorities

    def forward_critic(self, state, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(state, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.critic_gru(packed) # hidden: (num_layers * num_directions, batch, hidden_size)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)  # hidden[0]: 정방향 은닉상태 [1]: 역방향 은닉상태, 이 둘을 dim=1 방향으로 연결, hidden_cat:(batch, hidden_size * 2)
        value = self.critic_fc(hidden_cat).squeeze(-1)  # (batch,)
        return value

    def masked_softmax(self, priorities, mask):
        mask = mask.to(dtype=torch.bool) if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool)
        priorities = priorities.masked_fill(mask == 0, float('-inf'))
        probs = F.softmax(priorities, dim=1)
        return probs

    def act(self, state, lengths, available_jobs_mask):
        # state: (batch, seq_len, input_size)
        priorities = self.forward_actor(state, lengths)  # (batch, seq_len)
        probs = self.masked_softmax(priorities, available_jobs_mask)  # 마스킹 적용 후 Softmax 정규화
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs

class Two_Stage_RNN:
    hidden_size = 32
    fc_size = 64
    learning_rate = 0.0001
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.3
    num_first_episodes = 1500
    num_second_episodes = 4500
    update_steps = 80
    batch_size = 9
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
        return torch.tensor(batch_states, dtype=torch.float32), lengths

    @classmethod
    def train_first_stage(cls, model, optimizer, memory, loss_list):
        states, actions, rewards, log_probs_old, values, next_values, dones = zip(*memory)

        states, lengths = cls.pad_states(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).detach()
        values = torch.tensor(values, dtype=torch.float32).detach()
        next_values = torch.tensor(next_values, dtype=torch.float32).detach()
        dones = torch.tensor(dones, dtype=torch.float32)

        # TD-Target 계산
        td_target = rewards + cls.gamma * next_values * (1 - dones)
        # TD Error 계산
        delta = (td_target - values).detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for t in reversed(range(len(delta))):
            # if dones[t] == 1:
            #     advantage = 0
            advantage = cls.gamma * cls.gae_lambda * advantage + delta[t]
            advantage_lst.append([advantage])
        advantage_lst.reverse()

        # Advantage 값을 PyTorch Tensor로 변환 & 정규화
        advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float32)
        advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(cls.update_steps):
            _, _, action_probs = cls.model.act(states, lengths, available_jobs_mask=torch.ones_like(states[..., 0]))
            log_pi_all = F.log_softmax(action_probs, dim=-1)

            # 선택된 행동에 대한 log π(a_t|s_t)
            log_pi = log_pi_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(log_pi - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cls.clip_epsilon, 1 + cls.clip_epsilon) * advantages
            entropy = -(action_probs * log_pi_all).sum(dim=1).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - cls.entropy_coef * entropy

            values_pred = cls.model.forward_critic(states, lengths)  # 최신 Critic 네트워크의 예측값
            critic_loss = F.smooth_l1_loss(values_pred, td_target)  # 예측값과 실제 목표값(returns) 비교하여 Loss 계산

            loss = actor_loss + critic_loss
            loss_list.append(loss.item())

            cls.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Gradient Clipping 추가
            cls.optimizer.step()

    @classmethod
    def train_second_stage(cls, model, optimizer, memory, loss_list):
        states_list, actions_list, rewards_list, log_probs_old_list, done_mask_list = [], [], [], [], []

        for episode in memory:
            states, actions, rewards, log_probs_old, done_mask = zip(*episode)
            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            log_probs_old_list.append(log_probs_old)
            done_mask_list.append(done_mask)

        # Flatten all episodes into one big batch
        flat_states = [s for episode in states_list for s in episode]
        flat_actions = [a for episode in actions_list for a in episode]
        flat_rewards = [r for episode in rewards_list for r in episode]
        flat_log_probs_old = [lp for episode in log_probs_old_list for lp in episode]
        flat_done_mask = [d for episode in done_mask_list for d in episode]

        states, lengths = cls.pad_states(flat_states)
        actions = torch.tensor(flat_actions, dtype=torch.long)
        rewards = torch.tensor(flat_rewards, dtype=torch.float32)
        log_probs_old = torch.tensor(flat_log_probs_old, dtype=torch.float32).detach()
        done_mask = torch.tensor(flat_done_mask, dtype=torch.float32)

        # Monte Carlo 방식으로 Return 계산
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(done_mask)):
            G = reward + cls.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Advantage 계산
        with torch.no_grad():
            state_values = cls.model.forward_critic(states, lengths)
        advantage = returns - state_values
        advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(cls.update_steps):
            _, _, action_probs = cls.model.act(states, lengths, available_jobs_mask=torch.ones_like(states[..., 0]))
            log_action_probs = torch.log(action_probs)
            log_pi = log_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(log_pi - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cls.clip_epsilon, 1 + cls.clip_epsilon) * advantages
            entropy = -(action_probs * log_action_probs).sum(dim=1).mean()

            actor_loss = -torch.min(surr1, surr2).mean() - cls.entropy_coef * entropy

            values_pred = cls.model.forward_critic(states, lengths)
            critic_loss = F.smooth_l1_loss(values_pred, returns)

            loss = actor_loss + critic_loss
            loss_list.append(loss.item())

            cls.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cls.model.parameters(), max_norm=0.5)
            cls.optimizer.step()

    @classmethod
    def main(cls):
        cls.env = Simulator()
        cls.model = RNN_ActorCritic(input_size=6, hidden_size=cls.hidden_size, fc_size=cls.fc_size)
        cls.optimizer = optim.Adam(cls.model.parameters(), lr=cls.learning_rate)

        first_train_makespan_list, first_train_score_list, first_train_tardiness_list, first_train_loss_list = [], [], [], []
        first_train_last_20_makespan, first_train_last_20_score, first_train_last_20_tardiness = [], [], []

        second_train_makespan_list, second_train_score_list, second_train_tardiness_list, second_train_loss_list = [], [], [], []
        second_train_last_20_makespan, second_train_last_20_score, second_train_last_20_tardiness = [], [], []

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)

        cls.plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(cls.plot_save_directory, exist_ok=True)

        first_score, second_score = 0.0, 0.0
        for n_epi in range(cls.num_first_episodes):
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                first_score = 0.0
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    available_job = cls.env.available_jobs_mask()
                    action, log_prob, _ = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    value = cls.model.forward_critic(state_tensor, [state.shape[0]])
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action.item(), n_epi)
                    done_mask = 1.0 if done else 0.0
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    next_value = cls.model.forward_critic(next_state_tensor, [next_state.shape[0]])  # 다음 상태의 value 계산
                    cls.memory_first_stage.append((state, action.item(), reward, log_prob.item(), value.item(), next_value.item(), done_mask))
                    if len(cls.memory_first_stage) >= cls.batch_size:
                        cls.train_first_stage(cls.model, cls.optimizer, cls.memory_first_stage, first_train_loss_list)
                        cls.memory_first_stage = []
                    state = next_state
                    first_score += reward
                    if done:
                        break

            first_train_makespan_list, first_train_score_list, first_train_tardiness_list = cls.script_performance(cls.env, n_epi, cls.memory_first_stage, first_score, first_train_makespan_list, first_train_score_list, first_train_tardiness_list)
            first_train_last_20_makespan.append(np.mean(first_train_makespan_list[-20:]))
            first_train_last_20_score.append(np.mean(first_train_score_list[-20:]))
            first_train_last_20_tardiness.append(np.mean(first_train_tardiness_list[-20:]))

            if Parameters.param_down_on:
                file_name = str(n_epi) + "param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(cls.model.state_dict(), file_path)

        object_name = 'Total Tardiness' if Hyperparameters.object == 'total_tardiness' else 'Makespan'
        first_train_data = first_train_tardiness_list if object_name == 'Total Tardiness' else first_train_makespan_list
        first_last_train_20 = first_train_last_20_tardiness if object_name == 'Total Tardiness' else first_train_last_20_makespan
        cls.plot_training_progress(object_name, 'First', first_train_data, first_last_train_20, first_train_score_list, first_train_last_20_score, first_train_loss_list)
        if Parameters.gantt_on_check:
            cls.env.gantt_chart(first_score)

        for n_epi in range(cls.num_first_episodes, cls.num_first_episodes + cls.num_second_episodes):
            for dataid in Parameters.datasetId:
                state = cls.env.reset(dataid)
                done = False
                second_score = 0.0
                episode_memory = []
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    available_job = cls.env.available_jobs_mask()
                    action, log_prob, _ = cls.model.act(state_tensor, [state.shape[0]], available_job)
                    value = cls.model.forward_critic(state_tensor, [state.shape[0]])
                    next_state, reward, done, _ = cls.env.two_stage_rnn(action.item(), n_epi)
                    done_mask = 1.0 if done else 0.0
                    episode_memory.append((state, action.item(), reward, log_prob.item(), done_mask))
                    state = next_state
                    second_score += reward
                    if done:
                        break
                cls.memory_second_stage.append(episode_memory)

            cls.train_second_stage(cls.model, cls.optimizer, cls.memory_second_stage, second_train_loss_list)
            cls.memory_second_stage = []

            second_train_makespan_list, second_train_score_list, second_train_tardiness_list = cls.script_performance(cls.env, n_epi, cls.memory_second_stage, second_score, second_train_makespan_list, second_train_score_list, second_train_tardiness_list)
            second_train_last_20_makespan.append(np.mean(second_train_makespan_list[-20:]))
            second_train_last_20_score.append(np.mean(second_train_score_list[-20:]))
            second_train_last_20_tardiness.append(np.mean(second_train_tardiness_list[-20:]))

            if Parameters.param_down_on:
                file_name = str(n_epi) + "param.pt"
                file_path = os.path.join(save_directory, file_name)
                torch.save(cls.model.state_dict(), file_path)

        # second training 그래프
        object_name = 'Total Tardiness' if Hyperparameters.object == 'total_tardiness' else 'Makespan'
        second_train_data = second_train_tardiness_list if object_name == 'Total Tardiness' else second_train_makespan_list
        second_last_train_20 = second_train_last_20_tardiness if object_name == 'Total Tardiness' else second_train_last_20_makespan
        cls.plot_training_progress(object_name, 'Second', second_train_data, second_last_train_20, second_train_score_list, second_train_last_20_score, second_train_loss_list)
        if Parameters.gantt_on_check:
            cls.env.gantt_chart(second_score)

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
            print(output_string)
            logging.info(f'performance :{output_string}')
        return makespan_list, score_list, tardiness_list

    @classmethod
    def plot_training_progress(cls, object_name, training_stage, train_data, last_train_20, score_list, last_score_20, loss_list):
        fig, axs = plt.subplots(3, 2, figsize=(58, 26))

        axs[0, 0].plot(range(len(train_data)), train_data, linestyle='--', marker='.', color='green', label=f'Training {object_name}')
        axs[0, 0].set_xlabel('Episodes', fontsize=32)
        axs[0, 0].set_ylabel(f'{training_stage} Training {object_name}', fontsize=32)
        axs[0, 0].tick_params(axis='both', labelsize=28)

        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='blue', label='Training Score')
        axs[1, 0].set_xlabel('Episodes', fontsize=32)
        axs[1, 0].set_ylabel(f'{training_stage} Training Score', fontsize=32)
        axs[1, 0].tick_params(axis='both', labelsize=28)

        axs[0, 1].plot(range(len(last_train_20)), last_train_20, linestyle='--', marker='.', color='green', label=f'Training {object_name}')
        axs[0, 1].set_xlabel('Episodes', fontsize=32)
        axs[0, 1].set_ylabel(f'Average {object_name} over previous 50 episodes', fontsize=32)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        axs[1, 1].plot(range(len(last_score_20)), last_score_20, linestyle='--', marker='.', color='blue', label='Training Score')
        axs[1, 1].set_xlabel('Episodes', fontsize=32)
        axs[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=32)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        axs[2, 0].plot(range(len(loss_list)), loss_list, color='red', label='Training Loss')
        axs[2, 0].set_xlabel('Episodes', fontsize=32)
        axs[2, 0].set_ylabel(f'{training_stage} Training Loss', fontsize=32)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        correlation = np.corrcoef(train_data, score_list)[0, 1]
        axs[2, 1].scatter(train_data, score_list, color='black', marker='o', label=f'Correlation: {correlation:.2f}')
        axs[2, 1].set_xlabel(object_name, fontsize=32)
        axs[2, 1].set_ylabel('Score', fontsize=32)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        axs[2, 1].legend(fontsize=28)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name_1 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_{training_stage}_train_result.png"
            file_path_1 = os.path.join(cls.plot_save_directory, result_name_1)
            fig.savefig(file_path_1)  # Training plot 저장
            print(f"첫 번째 그래프가 저장되었습니다: {file_path_1}")
        plt.show()