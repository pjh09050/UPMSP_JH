import numpy as np
from src.learner.common.Hyperparameters import *
from src.simlator.Simulator import *
import math

class RewardManager:
    @classmethod
    def get_setup_pm_d_time(cls, machine, j_list, r_list, curr_time, candidate, candidate_list, finish_job, phase):
        # phase("first"/"second")와 무관하게 같은 보상: PM은 연속 PM 횟수만큼 PM 시간, job은 열화로 늘어난 시간 + setup 시간
        r = 0
        if candidate == None:
            return 0, r_list
        if candidate[0] == "Maintenance":
            pm_time = Hyperparameters.maintenance_time
            pm_count = r_list[machine].continuous_maintenance_count
            r -= pm_time * pm_count
        else:
            deterioration_time = candidate[1] - candidate[0].previous_p_time
            setup_time = candidate[2]
            r -= deterioration_time + setup_time
        return r, r_list

    @classmethod
    def get_setup(cls, r_list, candidate):
        if candidate == None:
            return 0, r_list
        if candidate[0] == "Maintenance":
            # return -Hyperparameters.maintenance_time, r_list
            return 0, r_list
        if candidate[2] == 0:
            r = 0
        else:
            # r = -candidate[2]
            r = -1
        return r, r_list

    @classmethod
    def two_stage_rnn(cls, machine, j_list, r_list, curr_time, candidate_list, candidate, finish_job, episode):
        if episode < 500:
            if candidate[2] == 0 and candidate[0] != "Maintenance":
                r = 1
            elif candidate[0] == "Maintenance":
                r = -1
            else:
                family_check = False
                for i in candidate_list:
                    if i[0] == "Maintenance":
                        continue  # Maintenance는 제외하고 반복
                    if r_list[machine].previous_family_type == i[0].family_type:
                        family_check = True
                        break
                if family_check:
                    r = -1
                else:
                    r = 0
        else:
            if finish_job == True:
                if Hyperparameters.object == 'total_tardiness':
                    total_tardiness = 0
                    for i in j_list:
                        tardiness = j_list[i].tardiness_time
                        total_tardiness += tardiness
                    if total_tardiness == 0:
                        r = -total_tardiness + 200
                    else:
                        r = -total_tardiness
                else:
                    makespan = max(r_list.values(), key=lambda machine: machine.reservation_time).reservation_time
                    r = -makespan
            else:
                r = 0
        return r, r_list

    @classmethod
    def get_reward(cls, reward_type, machine, j_list, r_list, curr_time, candidate_list, candidate, finish_job, episode):
        if reward_type == 'setup_pm_d_time':
            r, r_list = cls.get_setup_pm_d_time(machine, j_list, r_list, curr_time, candidate, candidate_list, finish_job, episode)
        elif reward_type == "setup":
            r, r_list = cls.get_setup(r_list, candidate)
        elif reward_type == "two_stage_rnn":
            r, r_list = cls.two_stage_rnn(machine, j_list, r_list, curr_time, candidate_list, candidate, finish_job, episode)
        return np.round(r, decimals=3), r_list

    @classmethod
    def get_simulator_instance(cls):
        from src.simlator.Simulator import Simulator
        cls.sim = Simulator()
        return cls.sim