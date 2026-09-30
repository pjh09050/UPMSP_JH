import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from src.simlator.Simulator import *
import matplotlib.pyplot as plt
from src.common.Parameters import *
np.set_printoptions(linewidth=500)  # Adjust the linewidth to your desired number

class PPO_Policy_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO_Policy_Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x, mask):
        logits = self.network(x)
        action_probs = self.masked_softmax(logits, mask) # 마스킹 적용 후 Softmax
        return action_probs

    def pi_log_prob(self, x, env, softmax_dim=-1):
        available_mask = env.get_valid_actions(x) # 가능한 행동 확인
        logits = self.network(x)
        action_probs = self.masked_softmax(logits, available_mask) # 마스킹 적용 후 Softmax
        log_pi = F.log_softmax(action_probs, dim=softmax_dim) # 로그 확률로 변환
        return action_probs, log_pi

    def masked_softmax(self, action_probs, mask):
        mask = mask.to(dtype=torch.bool) if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool)
        action_probs = action_probs.masked_fill(mask == 0, float('-inf')) # 불가능한 행동에 -inf 적용
        probs = F.softmax(action_probs, dim=-1) # Softmax 적용하여 확률 변환
        return probs

class PPO_Value_Network(nn.Module):
    def __init__(self, input_dim):
        super(PPO_Value_Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        value = self.network(x)
        return value

class PMS_PPO_batch:
    memory = []
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2 # 값이 클수록 정책이 과도하게 변형될 수 있음
    k_epoch = 10
    batch_size = 32
    max_grad_norm = 1.0
    policy_lr = 0.00005
    value_lr = 0.00001
    entropy_coef = 0.01 # 값이 클수록 탐험이 많아짐
    train_algorithm = "MC" # MC or TD

    @classmethod
    def select_action(cls, state, available_family_mask):
        state = torch.from_numpy(state).float()
        action_probs = cls.policy_network(state, available_family_mask) # 마스킹한 action의 확률을 계산함
        probs = cls.masked_softmax(action_probs, available_family_mask) # 불가능한 액션 마스킹 후 Softmax
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, probs

    @classmethod
    def select_best_action(cls, state, available_family_mask):
        state = torch.from_numpy(state).float()
        action_probs = cls.policy_network(state, available_family_mask) # 마스킹한 action의 확률을 계산함
        probs = cls.masked_softmax(action_probs, available_family_mask) # 마스킹 적용 후 Softmax
        return probs

    @classmethod
    def masked_softmax(self, action_probs, mask):
        mask = mask.to(dtype=torch.bool) if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool)
        action_probs = action_probs.masked_fill(mask == 0, float('-inf')) # 마스킹이 (0,1)이여서 0인 부분 -inf 처리
        probs = F.softmax(action_probs, dim=-1) # 마스킹된 확률에 대해 softmax 적용
        return probs

    @classmethod
    def store_transition(cls, transition):
        cls.memory.append(transition)

    @classmethod
    def make_batch(cls):
        s_lst, a_lst, r_lst, s_prime_lst, log_a_lst, done_lst = [], [], [], [], [], []
        for transition in cls.memory:
            s, a, r, s_prime, log_a, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            log_a_lst.append([log_a])
            done_lst.append([done_mask])

        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst), dtype=torch.int64)
        r = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        log_a = torch.tensor(np.array(log_a_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)

        # 랜덤 샘플링을 위한 배치 인덱스 생성
        num_samples = s.size(0)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        batch_indices = [indices[i:i + cls.batch_size] for i in range(0, num_samples, cls.batch_size)]

        return s, a, r, s_prime, log_a, done_mask, batch_indices

    @classmethod
    def train_gae(cls, policy_loss_list, value_loss_list):
        s, a, r, s_prime, old_log_pi, done_mask, batch_indices = cls.make_batch()

        # TD-Target 계산
        td_target = r + cls.gamma * cls.value_network(s_prime) * (1 - done_mask)
        # TD Error 계산
        delta = (td_target - cls.value_network(s)).detach().numpy()

        # GAE 사용하여 Advantage 계산
        advantage_lst = []
        advantage = 0.0
        for t in reversed(range(len(delta))):
            if done_mask[t] == 1:
                advantage = 0
            advantage = cls.gamma * cls.gae_lambda * advantage + delta[t][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()

        # Advantage 값을 PyTorch Tensor로 변환 & 정규화
        advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(cls.k_epoch):
            for batch_idx in batch_indices:
                # 현재 정책 (`π_θ`)에서 행동 확률 및 로그 확률 계산
                action_probs, log_pi_all = cls.policy_network.pi_log_prob(s[batch_idx], cls.env, softmax_dim=1)

                # 선택된 행동에 대한 log π(a_t|s_t)
                log_pi = log_pi_all.gather(1, a[batch_idx]).squeeze()

                # 확률 비율 (π_θ / π_old)
                ratio = torch.exp(log_pi - old_log_pi[batch_idx])

                # PPO Clipping 적용
                surr1 = ratio * advantage[batch_idx]
                surr2 = torch.clamp(ratio, 1 - cls.eps_clip, 1 + cls.eps_clip) * advantage[batch_idx]

                # 엔트로피 계산
                entropy = -(action_probs * log_pi_all).sum(dim=1).mean()

                # Policy Network 학습 (Actor Update)
                policy_loss = -torch.min(surr1, surr2).mean() - cls.entropy_coef * entropy

                # Value Network 학습 (Critic Update)
                state_values = cls.value_network(s[batch_idx])
                value_loss = F.smooth_l1_loss(state_values, td_target[batch_idx].detach())

                # 총 손실 = policy_loss + value_loss
                total_loss = policy_loss + value_loss

                # 손실 기록
                policy_loss_list.append(policy_loss.item())
                value_loss_list.append(value_loss.item())

                # 전체 네트워크 업데이트
                cls.policy_optimizer.zero_grad()
                cls.value_optimizer.zero_grad()
                total_loss.backward()

                # 두 네트워크에 대해 따로 gradient clipping & step
                torch.nn.utils.clip_grad_norm_(cls.policy_network.parameters(), max_norm=cls.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(cls.value_network.parameters(), max_norm=cls.max_grad_norm)
                cls.policy_optimizer.step()
                cls.value_optimizer.step()
        cls.memory.clear()

    @classmethod
    def train_mc(cls, policy_loss_list, value_loss_list):
        s, a, r, s_prime, old_log_pi, done_mask, _ = cls.make_batch()  # batch_indices 제거

        # G 계산을 위해 reward와 done mask를 리스트로 변환
        r = r.squeeze().tolist()  # shape: [N]
        done_mask = done_mask.squeeze().tolist()  # shape: [N]

        # Monte Carlo 방식으로 Return G 계산
        returns = []
        G = 0
        for reward, done in zip(reversed(r), reversed(done_mask)):
            G = reward + cls.gamma * G * (1 - done)  # done이면 G 초기화
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Advantage = G - V(s)
        with torch.no_grad():
            state_values = cls.value_network(s).squeeze()
        advantage = returns - state_values
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(cls.k_epoch):
            # 현재 정책에서 행동 로그 확률 및 확률 계산
            action_probs, log_pi_all = cls.policy_network.pi_log_prob(s, cls.env, softmax_dim=-1)

            # 선택된 행동에 대한 log π(a_t|s_t)
            log_pi = log_pi_all.gather(1, a).squeeze()

            # π_θ / π_old 비율 계산
            ratio = torch.exp(log_pi - old_log_pi.squeeze())

            # PPO clipping 적용
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - cls.eps_clip, 1 + cls.eps_clip) * advantage
            entropy = -(action_probs * log_pi_all).sum(dim=1).mean()

            # Policy Loss (Clipped PPO + Entropy 보너스)
            policy_loss = -torch.min(surr1, surr2).mean() - cls.entropy_coef * entropy

            # Value Loss: G와 V(s)의 차이
            value_pred = cls.value_network(s).squeeze()
            value_loss = F.smooth_l1_loss(value_pred, returns.detach())

            # 전체 손실
            total_loss = policy_loss + value_loss

            # 손실 기록
            policy_loss_list.append(policy_loss.item())
            value_loss_list.append(value_loss.item())

            # 전체 네트워크 업데이트
            cls.policy_optimizer.zero_grad()
            cls.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(cls.policy_network.parameters(), max_norm=cls.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(cls.value_network.parameters(), max_norm=cls.max_grad_norm)
            cls.policy_optimizer.step()
            cls.value_optimizer.step()
        cls.memory.clear()

    @classmethod
    def main(cls):
        cls.env = Simulator()
        cls.policy_network = PPO_Policy_Network(Hyperparameters.input_layer, Hyperparameters.output_layer)
        cls.value_network = PPO_Value_Network(Hyperparameters.input_layer)
        cls.policy_optimizer = optim.Adam(cls.policy_network.parameters(), lr=cls.policy_lr)
        cls.value_optimizer = optim.Adam(cls.value_network.parameters(), lr=cls.value_lr)

        save_directory = f"{pathConfig.model_save_path}{os.sep}{Parameters.simulation_time}_{Parameters.datasetId}"
        if Parameters.param_down_on:
            os.makedirs(save_directory, exist_ok=True)  # 경로 없을 시 생성

        plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
        if Parameters.save_plot:
            os.makedirs(plot_save_directory, exist_ok=True)

        score_list = []
        makespan_list = []
        policy_loss_list = []
        value_loss_list = []
        last_50_score = []
        last_50_makespan = []
        learn_iter = 0
        time_step = 0

        test_score_list = []
        test_makespan_list = []
        test_last_50_score = []
        test_last_50_makespan = []

        for n_epi in range(Hyperparameters.episode):
            for dataid in Parameters.datasetId:
                score = 0
                s = cls.env.reset(dataid)
                done = False
                while not done:
                    # print('s', s)
                    available_family = cls.env.available_family_mask() # 가능한 family type 여부 확인
                    action, log_prob, probs = cls.select_action(s, available_family)
                    s_prime, reward, done = cls.env.ppo_job_select(action, type='train')
                    done_mask = 1.0 if done else 0.0
                    cls.store_transition((s, action, reward, s_prime, log_prob.item(), done_mask))
                    # print('probs', probs)
                    # print('max action', torch.argmax(probs).item())
                    # print('a', action)
                    # print('r', reward)
                    # print('s_prime', s_prime)
                    # print('log_prob', log_prob.item())
                    # print('done', done)
                    # print('done_mask', done_mask)
                    # print('------------------------------------------------------')
                    s = s_prime
                    score += reward
                    time_step += 1
                    if cls.train_algorithm == "TD":
                        if time_step % cls.batch_size == 0: # batch size만큼의 step이 지나면 train을 수행함
                            cls.train_gae(policy_loss_list, value_loss_list)
                            learn_iter += 1
                    if done:
                        break

            if cls.train_algorithm == "MC":
                cls.train_mc(policy_loss_list, value_loss_list)
                learn_iter += 1
            score_list, makespan_list = cls.script_performance(cls.env, n_epi, score, dataid, score_list, makespan_list)
            if Parameters.log_on:
                print(f'Episode {n_epi}, time_steps: {time_step}, learn count: {learn_iter}, Score: {score:.3f}, Makespan: {makespan_list[-1]:.3f}')
            last_50_score.append(np.round(np.mean(score_list[-50:]), 3))
            last_50_makespan.append(np.round(np.mean(makespan_list[-50:]), 3))

            if Parameters.param_down_on:
                policy_file_name = str(n_epi) + "_policy_param.pt"
                policy_file_path = os.path.join(save_directory, policy_file_name)
                value_file_name = str(n_epi) + "_value_param.pt"
                value_file_path = os.path.join(save_directory, value_file_name)
                torch.save(cls.policy_network.state_dict(), policy_file_path)
                torch.save(cls.value_network.state_dict(), value_file_path)

            test = False
            # if n_epi >= 2500:
            #     test = True
            if test == True:
                for dataid in Parameters.datasetId:
                    score = 0
                    s = cls.env.reset(dataid)
                    done = False
                    while not done:
                        available_family = cls.env.available_family_mask()
                        probs = cls.select_best_action(s, available_family)
                        print('probs', probs)
                        action = torch.argmax(probs).item()  # 확률이 가장 높은 action 선택
                        print('action', action)
                        s_prime, reward, done = cls.env.ppo_job_select(action, type='test')
                        s = s_prime
                        score += reward
                    test_score_list, test_makespan_list = cls.script_performance(cls.env, n_epi, score, dataid, test_score_list, test_makespan_list)
                    test_avg_score = np.round(np.mean(test_score_list[-50:]), 3)
                    test_avg_makespan = np.round(np.mean(test_makespan_list[-50:]), 3)
                    print(f'Test Episode {n_epi}, time_steps: {time_step}, learn count: {learn_iter}, Test Score: {score:.3f}, Test Makespan: {test_makespan_list[-1]:.3f}')
                    print('=============================================================================')
                    test_last_50_score.append(test_avg_score)
                    test_last_50_makespan.append(test_avg_makespan)

        object = Hyperparameters.object
        fig, axs = plt.subplots(3, 2, figsize=(58, 26))
        axs[0, 0].plot(range(len(makespan_list)), makespan_list, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 0].set_xlabel('Episodes', fontsize=36)
        axs[0, 0].set_ylabel(f'Training {object}', fontsize=36)
        axs[0, 0].tick_params(axis='both', labelsize=28)

        axs[1, 0].plot(range(len(score_list)), score_list, linestyle='-', marker='.', color='blue', label='Training Score')
        axs[1, 0].set_xlabel('Episodes', fontsize=36)
        axs[1, 0].set_ylabel('Training Score', fontsize=36)
        axs[1, 0].tick_params(axis='both', labelsize=28)

        axs[0, 1].plot(range(len(last_50_makespan)), last_50_makespan, linestyle='--', marker='.', color='green', label=f'Training {object}')
        axs[0, 1].set_xlabel('Episodes', fontsize=36)
        axs[0, 1].set_ylabel(f'Average {object} over previous 50 episodes', fontsize=36)
        axs[0, 1].tick_params(axis='both', labelsize=28)

        axs[1, 1].plot(range(len(last_50_score)), last_50_score, linestyle='--', marker='.', color='blue', label=f'Training Score')
        axs[1, 1].set_xlabel('Episodes', fontsize=36)
        axs[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=36)
        axs[1, 1].tick_params(axis='both', labelsize=28)

        axs[2, 0].plot(range(len(policy_loss_list)), policy_loss_list, color='red', label='Training Policy Loss')
        axs[2, 0].set_xlabel('Episodes', fontsize=36)
        axs[2, 0].set_ylabel('Training Policy Loss', fontsize=36)
        axs[2, 0].tick_params(axis='both', labelsize=28)

        axs[2, 1].plot(range(len(value_loss_list)), value_loss_list, color='red', label='Training Value Loss')
        axs[2, 1].set_xlabel('Episodes', fontsize=36)
        axs[2, 1].set_ylabel('Training Value Loss', fontsize=36)
        axs[2, 1].tick_params(axis='both', labelsize=28)
        plt.tight_layout()

        if test == True:
            fig1, axs1 = plt.subplots(2, 2, figsize=(48, 24))
            axs1[0, 0].plot(range(len(test_makespan_list)), test_makespan_list, linestyle='--', marker='.', color='green', label=f'Test {object}')
            axs1[0, 0].set_xlabel('Test Episodes', fontsize=36)
            axs1[0, 0].set_ylabel(f'Test {object}', fontsize=36)
            axs1[0, 0].tick_params(axis='both', labelsize=28)

            axs1[1, 0].plot(range(len(test_score_list)), test_score_list, linestyle='-', marker='.', color='blue', label='Test Score')
            axs1[1, 0].set_xlabel('Test Episodes', fontsize=36)
            axs1[1, 0].set_ylabel('Test Score', fontsize=36)
            axs1[1, 0].tick_params(axis='both', labelsize=28)

            axs1[0, 1].plot(range(len(test_last_50_makespan)), test_last_50_makespan, linestyle='--', marker='.', color='green', label=f'Test {object}')
            axs1[0, 1].set_xlabel('Test Episodes', fontsize=36)
            axs1[0, 1].set_ylabel(f'Average {object} over previous 50 episodes', fontsize=36)
            axs1[0, 1].tick_params(axis='both', labelsize=28)

            axs1[1, 1].plot(range(len(test_last_50_score)), test_last_50_score, linestyle='--', marker='.', color='blue', label=f'Test Score')
            axs1[1, 1].set_xlabel('Test Episodes', fontsize=36)
            axs1[1, 1].set_ylabel(f'Average Score over previous 50 episodes', fontsize=36)
            axs1[1, 1].tick_params(axis='both', labelsize=28)
            plt.tight_layout()

        if Parameters.save_plot:
            result_name_1 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_train_result.png"
            file_path_1 = os.path.join(plot_save_directory, result_name_1)
            fig.savefig(file_path_1)  # Training plot 저장
            print(f"첫 번째 그래프가 저장되었습니다: {file_path_1}")

            # Test 그래프 저장
            if test == True:
                result_name_2 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_test_result.png"
                file_path_2 = os.path.join(plot_save_directory, result_name_2)
                fig1.savefig(file_path_2)  # Test plot 저장
                print(f"두 번째 그래프가 저장되었습니다: {file_path_2}")

        if Parameters.gantt_on_check:
            cls.env.gantt_chart(score, None)
        plt.show()
        print("학습이 종료되었습니다.")

    @classmethod
    def get_evaluate(cls, checkpoint_path, number_of_checkpoint, point_num, datasets):
        cls.env = Simulator()

        if not hasattr(cls, 'policy_network') or not hasattr(cls, 'value_network'):
            cls.policy_network = PPO_Policy_Network(Hyperparameters.input_layer, Hyperparameters.output_layer)
            cls.value_network = PPO_Value_Network(Hyperparameters.input_layer)

        file_list = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))]
        check_point_list = sorted(set(int(f.split('_')[0]) for f in file_list if f.endswith('_policy_param.pt')))

        mean_reward_by_checkpoint = {}
        max_reward_by_checkpoint = {}

        best_makespan = float('inf')
        best_checkpoint = None

        makespan_list = []
        score_list = []
        for check_point_number in check_point_list[-point_num:]:
            policy_checkpoint = os.path.join(checkpoint_path, f"{check_point_number}_policy_param.pt")
            value_checkpoint = os.path.join(checkpoint_path, f"{check_point_number}_value_param.pt")

            policy_params = torch.load(policy_checkpoint)
            value_params = torch.load(value_checkpoint)
            cls.policy_network.load_state_dict(policy_params)
            cls.value_network.load_state_dict(value_params)

            cls.policy_network.eval()
            cls.value_network.eval()

            reward_list = []
            for dataset in datasets:
                s = cls.env.reset(dataset)
                done = False
                score = 0.0
                while not done:
                    available_family = cls.env.available_family_mask()  # 가능한 family type 여부 확인
                    probs = cls.select_best_action(s, available_family)
                    action = torch.argmax(probs).item()  # 확률이 가장 높은 action 선택
                    # print('a', action)
                    s_prime, reward, done = cls.env.ppo_job_select(action, type='test')
                    s = s_prime
                    score += reward
                    if done:
                        break

                Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = cls.env.performance_measure()
                reward_list.append(np.round(score, 3))
                makespan_list.append(np.round(makespan, 3))
                score_list.append(np.round(score, 3))

                if makespan <= best_makespan:
                    best_makespan = makespan
                    best_checkpoint = check_point_number

                    print(f"checkpoint: {check_point_number}")
                    print(f"dataset: {dataset}")
                    print(f"score:{score}")

                # print(f"checkpoint: {check_point_number}")
                # print(f"dataset: {dataset}")
                # print(f"score: {score}")

            mean_reward_by_checkpoint[check_point_number] = sum(reward_list) / len(reward_list)
            max_reward_by_checkpoint[check_point_number] = max(reward_list)

        max_check_point = max(max_reward_by_checkpoint.items(), key=lambda x: (x[1], x[0]))
        mean_check_point = max(mean_reward_by_checkpoint.items(), key=lambda x: (x[1], x[0]))

        print(f"max_checkpoint: {max_check_point}")
        print(f"mean_checkpoint: {mean_check_point}")
        print(f"Best Makespan: {best_makespan} with checkpoint: {best_checkpoint}")
        print("평가가 종료되었습니다.")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))  # 2행 1열, x축 공유
        x = list(range(point_num, point_num + len(score_list)))

        ax1.plot(x, makespan_list, linestyle='--', marker='.', label='Makespan', color='blue')
        ax1.set_xlabel('Episodes', fontsize=18)
        ax1.set_ylabel('Makespan', fontsize=18)

        ax2.plot(x, score_list, linestyle='--', marker='.', label='Score', color='green')
        ax2.set_xlabel('Episodes', fontsize=18)
        ax2.set_ylabel('Score', fontsize=18)
        plt.tight_layout()

        if Parameters.save_plot:
            plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
            os.makedirs(plot_save_directory, exist_ok=True)

            result_name_1 = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_evaluate_result.png"
            file_path_1 = os.path.join(plot_save_directory, result_name_1)
            plt.savefig(file_path_1)
            print(f"평가 그래프가 저장되었습니다: {file_path_1}")
        plt.show()

    @classmethod
    def get_result(cls, policy_parameter, value_parameter, dataSets):
        cls.env = Simulator()
        if not hasattr(cls, 'policy_network') or not hasattr(cls, 'value_network'):
            cls.policy_network = PPO_Policy_Network(Hyperparameters.input_layer, Hyperparameters.output_layer)
            cls.value_network = PPO_Value_Network(Hyperparameters.input_layer)

        policy_params = torch.load(policy_parameter)
        value_params = torch.load(value_parameter)
        cls.policy_network.load_state_dict(policy_params)
        cls.value_network.load_state_dict(value_params)

        cls.policy_network.eval()
        cls.value_network.eval()

        for data_id in dataSets:
            s = cls.env.reset(data_id)
            done = False
            score = 0.0
            while not done:
                print('s', s)
                available_family = cls.env.available_family_mask()  # 가능한 family type 여부 확인
                probs = cls.select_best_action(s, available_family)
                print('probs', probs)
                action = torch.argmax(probs).item()  # 확률이 가장 높은 action 선택
                s_prime, reward, done = cls.env.ppo_job_select(action, type='test')
                print('a', action)
                print('r', reward)
                print('s_prime', s_prime)
                print('done', done)
                print('------------------------------------------------------')
                s = s_prime
                score += reward
                if done:
                    break

            Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = cls.env.performance_measure()
            cls.env.gantt_chart(score, None)

            print(f"dataset: {data_id}")
            print(f"score: {score}")
            print("makespan:", round(makespan, 3))
            print("평가가 종료되었습니다.")

    @classmethod
    def script_performance(cls, env, n_epi, score, dataid, reward_list, makespan_list):
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = env.performance_measure()
        output_string = f"n_episode: {n_epi}, score : {score}\n" + \
                        f"Total Flow time: {Total_Flow_time:.2f}, util : {util:.2f}, makespan : {makespan:.2f}\n" + \
                        f"Total Lateness : {Total_Lateness_time:.2f}, Total Tardiness: {Total_Tardiness_time:.2f}, T_max: {T_max:.2f}\n" + \
                        f"--------------------------------------------------------------------------------------"
        if Parameters.log_on:
            # print(output_string)
            logging.info(f'performance :{output_string}')
        reward_list.append(score)
        makespan_list.append(makespan)
        return reward_list, makespan_list