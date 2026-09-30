import random
from src.learner.common.Hyperparameters import *
from src.simlator.dispatcher import *
from src.master_db.DataInventory import *
import torch
from torch.distributions import Categorical
from src.simlator.Simulator import *
import numpy as np
np.set_printoptions(linewidth=300)  # Adjust the linewidth to your desired number


class ActionManager:
    action_type = ""
    setup_type_list = []
    action_count = ""
    action_dimension = []
    sim = None

    @classmethod
    def get_lot(cls, candidate_list, action, current_time, machineId, machine_list, eps, type):
        rule_name = None
        if Hyperparameters.action_type == "dsp_rule":
            family_list = list(range(1, Hyperparameters.family_type + 1))
            filtered_candidates = [c for c in candidate_list if c[0] != "Maintenance"]
            candidate_to_family_type = sorted(set(candidate[0].family_type for candidate in filtered_candidates))
            missing_family_types = set(family_list) - set(candidate_to_family_type)
            valid_action_indices = []
            for idx, rule in enumerate(Hyperparameters.action_dimension):
                match = re.match(r"(\d+)_([A-Za-z]+)", rule)
                if match:
                    family_type = int(match.group(1))
                    if family_type not in missing_family_types:
                        valid_action_indices.append(idx)
            sim = cls.get_simulator_instance()
            if sum(machine_list[i].pm_count for i in machine_list) >= np.sqrt(sim.number_of_machine) * np.sqrt(sim.number_of_job):
                pass
            else:
                valid_action_indices.append(len(action)-1)
            coin = random.random()
            if coin < eps:
                action_idx = random.choice(valid_action_indices)
            else:
                q_values_valid = action[valid_action_indices]
                best_valid_idx = q_values_valid.argmax().item()
                action_idx = valid_action_indices[best_valid_idx]
            rule_name = Hyperparameters.action_dimension[action_idx]
            candidate = Dispatcher.dispatching_rule_decision(candidate_list, rule_name, machineId, machine_list)
            return str(rule_name), action_idx, candidate
        elif Hyperparameters.action_type =="job_select":
            candidate, action, setup = cls.job_select(candidate_list, action, eps)
            return candidate, action, setup
        elif Hyperparameters.action_type == "mc_job_select":
            candidate, action, setup = cls.mc_job_select(candidate_list, action, eps)
            return candidate, action, setup
        elif Hyperparameters.action_type == "two_stage_dqn" or Hyperparameters.action_type == "two_stage_dqn_rule":
            candidate, action, setup = cls.two_stage_dqn(machine_list, machineId, candidate_list, action, eps, type)
            return candidate, action, setup
        elif Hyperparameters.action_type == "ppo_job_select":
            candidate, action, setup, prob_a = cls.ppo_job_select(candidate_list, action, type)
            return candidate, action, setup, prob_a
        elif Hyperparameters.action_type == "two_stage_rnn":
            candidate, action, setup = cls.two_stage_rnn(candidate_list, action)
            return candidate, action, setup

    @classmethod
    def two_stage_dqn(cls, machine_list, machineId, candidate_list, action, eps, type):
        sim = cls.get_simulator_instance()
        a = sim.family_set
        if type == "first":
            if action == 0:
                pm_candidates = next((candidate for candidate in candidate_list if candidate[0] == "Maintenance"), None)
                return pm_candidates, action, 'PM'
            elif action == 1:
                current_machine_family = machine_list[machineId].family_type
                valid_action_list = []
                invalid_action_list = []
                for candidate in candidate_list:
                    if candidate[0] == "Maintenance":
                        continue
                    if candidate[0].family_type == current_machine_family:
                        valid_action_list.append(candidate)
                    else:
                        invalid_action_list.append(candidate)
                if valid_action_list:
                    best_candidate = random.choice(valid_action_list)
                else:
                    best_candidate = random.choice(invalid_action_list)
                return best_candidate, action - 1, str(best_candidate[0].family_type)
        elif type == "second":
            if Hyperparameters.action_type == "two_stage_dqn_rule":
                family_list = list(range(1, Hyperparameters.family_type + 1))
                candidate_to_family_type = sorted(set(candidate[0].family_type for candidate in candidate_list))
                missing_family_types = set(family_list) - set(candidate_to_family_type)
                valid_action_indices = []
                for idx, rule in enumerate(Hyperparameters.action_dimension):
                    match = re.match(r"(\d+)_([A-Za-z]+)", rule)
                    if match:
                        family_type = int(match.group(1))
                        if family_type not in missing_family_types:
                            valid_action_indices.append(idx)

                coin = random.random()
                if coin < eps:
                    action_idx = random.choice(valid_action_indices)
                else:
                    q_values_valid = action[valid_action_indices]
                    best_valid_idx = q_values_valid.argmax().item()
                    action_idx = valid_action_indices[best_valid_idx]
                rule_name = Hyperparameters.action_dimension[action_idx]
                candidate = Dispatcher.dispatching_rule_decision(candidate_list, rule_name, machineId, machine_list)
                return candidate, action_idx, str(rule_name)
            else:
                valid_action_list = action
                family_list = list(range(1, Hyperparameters.family_type + 1))
                candidate_to_family_type = sorted(set(candidate[0].family_type for candidate in candidate_list))
                missing_family_types = set(family_list) - set(candidate_to_family_type)
                invalid_action_indices = [family_list.index(f) for f in missing_family_types if f in family_list]
                valid_action_list = torch.tensor([a for i, a in enumerate(valid_action_list) if i not in invalid_action_indices])
                if eps == "None":
                    selected_family_type = max(zip(valid_action_list, candidate_to_family_type), key=lambda x: x[0])[1]
                else:
                    if random.random() < eps:
                        selected_family_type = random.choice(candidate_to_family_type)
                    else:
                        selected_family_type = max(zip(valid_action_list, candidate_to_family_type), key=lambda x: x[0])[1]
                filtered_candidates = [candidate for candidate in candidate_list if candidate[0].family_type == selected_family_type]
                # best_candidate = min(filtered_candidates,key=lambda can_job: np.round(machine_list[machineid].deterioration_rate[int(can_job[3][1:3]) - 1], decimals=3))
                best_candidate = min(filtered_candidates, key=lambda x: x[0].deterioration_rate)
                return best_candidate, selected_family_type - 1, str(selected_family_type)

    @classmethod
    def two_stage_rnn(cls, candidate_list, action):
        # if Hyperparameters.object == 'total_tardiness':
        #     same_lot = []
        #     for candidate in candidate_list:
        #         if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
        #             same_lot.append(candidate)
        #     min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
        #     return min_duedate_lot, action, setup
        # else:
        if Hyperparameters.maintenance == True:
            if action == len(Hyperparameters.action_dimension) - 1:
                for candidate in candidate_list:
                    if candidate[0] == "Maintenance":
                        return candidate, action, "Maintenance"
            else:
                for candidate in candidate_list:
                    if candidate[0] == "Maintenance":
                        pass
                    elif int(candidate[0].job_type[1:]) == action+1:
                        return candidate, action, f'J{action+1}'
        else:
            for candidate in candidate_list:
                if candidate[0] == "Maintenance":
                    pass
                elif int(candidate[0].job_type[1:]) == action + 1:
                    return candidate, action, f'J{action + 1}'

    @classmethod
    def mc_job_select(cls, candidate_list, action_list, eps):
        if Hyperparameters.maintenance == True:
            candidate_to_setup_list = [candidate[0].job_type if hasattr(candidate[0], 'job_type') else candidate[3] for candidate in candidate_list]
        else:
            candidate_to_setup_list = [candidate[0].job_type for candidate in candidate_list]  # setup list
        setup_index = [action for action, setup in enumerate(Hyperparameters.action_dimension) if setup in candidate_to_setup_list]
        if candidate_list is None or candidate_list == []:
            return None, len(action_list) - 1, "NOTHING"
        else:
            if eps == None :
                _, action, setup = max([action_list[i], i, Hyperparameters.action_dimension[i]] for i in setup_index)
            else:
                coin = random.random()
                if coin < eps:
                    candidate_action_list = []
                    for i in setup_index:
                        candidate_action_list.append([action_list[i], i, Hyperparameters.action_dimension[i]])
                    candidate_action = random.choice(candidate_action_list)
                    _, action, setup = candidate_action
                else:
                    _, action, setup = max([action_list[i], i, Hyperparameters.action_dimension[i]] for i in setup_index)
        if Hyperparameters.maintenance == True:
            if Hyperparameters.donothing == True:
                if Hyperparameters.object == 'total_tardiness':
                    if action == len(action_list) - 2:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        same_lot = []
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                same_lot.append(candidate)
                        min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                        return min_duedate_lot, action, setup
                else:
                    if action == len(action_list) - 2:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                return candidate, action, setup
            else:
                if Hyperparameters.object == 'total_tardiness':
                    if action == len(action_list) - 1:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        same_lot = []
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                same_lot.append(candidate)
                        min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                        return min_duedate_lot, action, setup
                else:
                    if action == len(action_list) - 1:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                return candidate, action, setup
        else:
            if Hyperparameters.object == 'total_tardiness':
                same_lot = []
                for candidate in candidate_list:
                    if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                        same_lot.append(candidate)
                min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                return min_duedate_lot, action, setup
            else:
                for candidate in candidate_list:
                    if candidate[0].job_type == setup:
                        return candidate, action, setup

    @classmethod
    def job_select(cls, candidate_list, action_list, eps):
        if Hyperparameters.maintenance == True:
            candidate_to_setup_list = [candidate[0].job_type if hasattr(candidate[0], 'job_type') else candidate[3] for candidate in candidate_list]
        else:
            candidate_to_setup_list = [candidate[0].job_type for candidate in candidate_list]  # setup list
        setup_index = [action for action, setup in enumerate(Hyperparameters.action_dimension) if setup in candidate_to_setup_list]
        if candidate_list is None or candidate_list == []:
            return None, (action_list.size(0) - 1 if isinstance(action_list, torch.Tensor) else len(action_list) - 1), "NOTHING"
        else:
            if eps == None :
                _, action, setup = max([action_list[i], i, Hyperparameters.action_dimension[i]] for i in setup_index)
            else:
                coin = random.random()
                if coin < eps:
                    candidate_action_list = []
                    for i in setup_index:
                        candidate_action_list.append([action_list[i], i, Hyperparameters.action_dimension[i]])
                    candidate_action = random.choice(candidate_action_list)
                    _, action, setup = candidate_action
                else:
                    _, action, setup = max([action_list[i], i, Hyperparameters.action_dimension[i]] for i in setup_index)
        if Hyperparameters.maintenance == True:
            if Hyperparameters.donothing == True:
                if Hyperparameters.object == 'total_tardiness':
                    if action == (action_list.size(0) - 2 if isinstance(action_list, torch.Tensor) else len(action_list) - 2):
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        same_lot = []
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                same_lot.append(candidate)
                        min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                        return min_duedate_lot, action, setup
                else:
                    if action == (action_list.size(0) - 2 if isinstance(action_list, torch.Tensor) else len(action_list) - 2):
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                return candidate, action, setup
            else:
                if Hyperparameters.object == 'total_tardiness':
                    if action == (action_list.size(0) - 1 if isinstance(action_list, torch.Tensor) else len(action_list) - 1):
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        same_lot = []
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                same_lot.append(candidate)
                        min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                        return min_duedate_lot, action, setup
                else:
                    if action == (action_list.size(0) - 1 if isinstance(action_list, torch.Tensor) else len(action_list) - 1):
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup
                    else:
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                return candidate, action, setup
        else:
            if Hyperparameters.object == 'total_tardiness':
                same_lot = []
                for candidate in candidate_list:
                    if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                        same_lot.append(candidate)
                min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                return min_duedate_lot, action, setup
            else:
                for candidate in candidate_list:
                    if candidate[0].job_type == setup:
                        return candidate, action, setup

    @classmethod
    def ppo_job_select(cls, candidate_list, action_prob, type):
        if Hyperparameters.maintenance == True:
            candidate_to_setup_list = [candidate[0].job_type if hasattr(candidate[0], 'job_type') else candidate[3] for candidate in candidate_list]
        else:
            candidate_to_setup_list = [candidate[0].job_type for candidate in candidate_list]  # setup list
        setup_index = [action for action, setup in enumerate(Hyperparameters.action_dimension) if setup in candidate_to_setup_list]

        def mask_invalid_jobs(probs, valid_indices):
            # masked_probs = probs.clone()  # 기존 확률을 복사
            masked_probs = probs
            invalid_indices = [i for i in range(len(probs)) if i not in valid_indices]
            masked_probs[invalid_indices] = 0  # 유효하지 않은 작업의 확률을 0으로 설정
            total_prob = masked_probs.sum()
            if total_prob > 0:
                masked_probs = masked_probs / total_prob
            else:
                masked_probs = torch.ones_like(masked_probs) / len(valid_indices)  # 균등 분포로 초기화
            return masked_probs

        if candidate_list is None or candidate_list == []:
            action = action_prob.size(0) - 1
            return None, action, "NOTHING", 1

        masked_action_probs = mask_invalid_jobs(action_prob, setup_index)
        # 정규화된 확률 기반으로 샘플링
        if type == 'train':
            m = Categorical(masked_action_probs)
            sampled_action = m.sample().item()  # 샘플링한 action index
            prob_a = masked_action_probs[sampled_action].item()
        else:
            sampled_action = torch.argmax(masked_action_probs).item()
            prob_a = masked_action_probs[sampled_action].item()

        # 실제 action 및 setup 값을 얻기 위해 setup_index를 사용
        action = sampled_action
        setup = Hyperparameters.action_dimension[action]
        if Hyperparameters.maintenance == True:
            if Hyperparameters.donothing == True:
                if Hyperparameters.object == 'total_tardiness':
                    if action == action_prob.size(0) - 2:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup, prob_a
                    else:
                        same_lot = []
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                same_lot.append(candidate)
                        min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                        return min_duedate_lot, action, setup, prob_a
                else:
                    if action == action_prob.size(0) - 2:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup, prob_a
                    else:
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                return candidate, action, setup, prob_a
            else:
                if Hyperparameters.object == 'total_tardiness':
                    if action == action_prob.size(0) - 1:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup, prob_a
                    else:
                        same_lot = []
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                same_lot.append(candidate)
                        min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                        return min_duedate_lot, action, setup, prob_a
                else:
                    if action == action_prob.size(0) - 1:
                        for candidate in candidate_list:
                            if isinstance(candidate[0], str) and candidate[0] == 'Maintenance':
                                return candidate, action, setup, prob_a
                    else:
                        for candidate in candidate_list:
                            if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                                return candidate, action, setup, prob_a
        else:
            if Hyperparameters.object == 'total_tardiness':
                same_lot = []
                for candidate in candidate_list:
                    if candidate[0] != 'Maintenance' and candidate[0].job_type == setup:
                        same_lot.append(candidate)
                min_duedate_lot = min(same_lot, key=lambda x: x[0].duedate)
                return min_duedate_lot, action, setup, prob_a
            else:
                for candidate in candidate_list:
                    if candidate[0].job_type == setup:
                        return candidate, action, setup, prob_a

    @classmethod
    def get_simulator_instance(cls):
        if cls.sim is None:
            from src.simlator.Simulator import Simulator
            cls.sim = Simulator()
        return cls.sim