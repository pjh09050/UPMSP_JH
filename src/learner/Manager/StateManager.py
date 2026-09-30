import numpy as np
from src.Object.Lot import *
from src.learner.common.Hyperparameters import *
from src.simlator.Simulator import *
import re


class StateManager:
    sim = None
    state_time = 0

    @classmethod
    def get_state(cls, j_list, r_list, cur_runtime, number_of_job, oper_in_list, candidate_list, candidate, machine_id, previous_machine_id):
        s = []
        if Hyperparameters.state_type == 'MC_DQN_state':
            s = cls.MC_DQN_state(j_list, r_list, cur_runtime, candidate_list, candidate, machine_id, previous_machine_id)
        elif Hyperparameters.state_type == 'two_stage_rnn':
            s = cls.two_stage_rnn_state(j_list, r_list, cur_runtime, candidate_list, machine_id)
        elif Hyperparameters.state_type == 'js_state':
            s = cls.js_state(j_list, r_list, cur_runtime, candidate_list, machine_id)
        return s

    @classmethod
    def MC_DQN_state(cls, j_list, r_list, cur_runtime, candidate_list, candidate, machine_id, previous_machine_id):
        s = []
        sim = cls.get_simulator_instance()

        family_type = Hyperparameters.family_type

        all_p_time, remain_p_time, done_p_time = cls.all_processing_time(sim, j_list, r_list)
        machine_sums = {machine: np.sum(times) for machine, times in all_p_time.items()}
        max_machine = max(machine_sums, key=machine_sums.get)
        max_time = machine_sums[max_machine]
        mean_processing_time = max_time / len(j_list)

        number_of_jobs_wait = 0
        wait_family_states = [0] * family_type
        wait_family_p_time = [0] * family_type
        wait_family_d_rate = [0] * family_type
        wait_family_counts = [0] * family_type  # 평균 계산용 개수 저장
        for i, job_id in enumerate(j_list):  # j_list에서 job_id로 job 접근
            job = j_list[job_id]  # 각 job의 정보 접근
            job_number = int(job.job_id[1:]) - 1
            if job.status == "WAIT":
                number_of_jobs_wait += 1
                f_type = job.family_type - 1
                if 0 <= f_type < family_type:
                    wait_family_states[f_type] += 1
                    wait_family_p_time[f_type] += job.processing_times[job.current_operation_id][machine_id]# 누적 처리시간
                    wait_family_d_rate[f_type] += np.round(r_list[machine_id].deterioration_rate[job_number], decimals=3)  # 누적 열화율
                    wait_family_counts[f_type] += 1

        s.extend([np.round(w / (len(j_list)/family_type), decimals=3) for w in wait_family_states])
        avg_wait_p_time = [
            np.round((wait_family_p_time[i] / wait_family_counts[i]) / mean_processing_time, 3) if wait_family_counts[i] > 0 else 0.0
            for i in range(family_type)
        ]
        avg_wait_d_rate = [
            np.round((wait_family_d_rate[i] / wait_family_counts[i]) / mean_processing_time, 3) if wait_family_counts[i] > 0 else 0.0
            for i in range(family_type)
        ]
        s.extend(avg_wait_p_time)
        s.extend(avg_wait_d_rate)

        # 현재 machine에 대한 정보
        machine_families = r_list[machine_id].family_type
        if isinstance(machine_families, str):
            machine_families = [int(fam) for fam in machine_families.split(',') if fam.strip().isdigit()] if machine_families != "NONE" else []
        elif isinstance(machine_families, int):
            machine_families = [machine_families]
        one_hot = [1 if j + 1 in machine_families else 0 for j in range(family_type)]
        s.extend(one_hot)

        machine_maintenance_count = np.round(r_list[machine_id].maintenance_count / (len(j_list)/len(r_list)), decimals=3)
        machine_pm_term = np.round(r_list[machine_id].pm_term / Hyperparameters.maintenance_time, decimals=3)
        # machine_maintenance_time = np.round((machine_maintenance_count * Hyperparameters.maintenance_time) / (max_time / len(j_list)), decimals=3)
        machine_quality = r_list[machine_id].machine_quality
        machine_deterioration_time = np.round(r_list[machine_id].deterioration_time / mean_processing_time, decimals=3)
        machine_setup_count = np.round(r_list[machine_id].setup_count / len(j_list), decimals=3)
        setup_time = 0 if candidate is None else candidate[2]
        machine_setup_time = np.round((machine_setup_count * setup_time) / mean_processing_time, decimals=3)
        # machine_cumsum_d_time = np.round((r_list[machine_id].machine_cumsum_d_time) / (max_time/len(j_list)), decimals=3)
        s.append(machine_maintenance_count)
        s.append(machine_pm_term)
        # s.append(machine_maintenance_time)
        s.append(machine_quality)
        s.append(machine_deterioration_time)
        # s.append(machine_cumsum_d_time)
        s.append(machine_setup_count)
        s.append(machine_setup_time)

        # 정규화 안하는 버전
        # s.append(r_list[machine_id].maintenance_count)
        # s.append(r_list[machine_id].machine_quality)
        # s.append(r_list[machine_id].deterioration_time)
        # s.append(r_list[machine_id].recovery_time)

        # # 이전 machine에 대한 정보
        # if previous_machine_id == None:
        #     # one_hot = [0 for j in range(Hyperparameters.family_type)]
        #     # s.extend(one_hot)
        #     s.append(0)
        #     s.append(0)
        #     s.append(0)
        # else:
        #     # machine_families = r_list[previous_machine_id].family_type
        #     # if isinstance(machine_families, str):
        #     #     machine_families = [int(fam) for fam in machine_families.split(',') if
        #     #                         fam.strip().isdigit()] if machine_families != "NONE" else []
        #     # elif isinstance(machine_families, int):
        #     #     machine_families = [machine_families]
        #     # one_hot = [1 if j + 1 in machine_families else 0 for j in range(Hyperparameters.family_type)]
        #     # s.extend(one_hot)
        #     previous_machine_maintenance_count = np.round(r_list[previous_machine_id].maintenance_count / len(r_list), decimals=3)
        #     previous_machine_quality = r_list[previous_machine_id].machine_quality
        #     previous_machine_deterioration_time = np.round(r_list[previous_machine_id].deterioration_time / len(j_list), decimals=3)
        #     s.append(previous_machine_maintenance_count)
        #     s.append(previous_machine_quality)
        #     s.append(previous_machine_deterioration_time)
        #     # 정규화 안하는 버전
        #     # s.append(r_list[previous_machine_id].maintenance_count)
        #     # s.append(r_list[previous_machine_id].machine_quality)
        #     # s.append(r_list[previous_machine_id].deterioration_time)

        # total_setup_number = 0
        # total_maintenance_number = 0
        # total_deterioration_time = 0
        # for i, resource in enumerate(r_list):
        #     total_setup_number += r_list[resource].setup_count
        #     total_maintenance_number += r_list[resource].maintenance_count
        #     total_deterioration_time += r_list[resource].deterioration_time
        #
        # s.append(np.round(total_setup_number / len(r_list), decimals=3))
        # s.append(np.round(total_maintenance_number / len(r_list), decimals=3))
        # s.append(np.round(total_deterioration_time / len(j_list) * len(r_list), decimals=3))

        # reward 미리 계산
        # if candidate == None:
        #     s.append(0)
        #     s.append(0)
        #     s.append(0)
        #     s.append(0)
        # else:
        #     r = 0
        #     if candidate[0] == "Maintenance":
        #         pm_time = Hyperparameters.maintenance_time
        #         r -= pm_time
        #         sim.total_pm_time_reward += pm_time
        #     else:
        #         deterioration_time = candidate[1] - candidate[0].previous_p_time
        #         setup_time = candidate[2]
        #         r -= setup_time + deterioration_time
        #         sim.total_setup_time_reward += setup_time
        #         sim.total_d_time_reward += deterioration_time
        #     s.append(sim.total_setup_time_reward)
        #     s.append(sim.total_d_time_reward)
        #     s.append(sim.total_pm_time_reward)
        #     s.append(sim.total_setup_time_reward + sim.total_d_time_reward + sim.total_pm_time_reward)

        df = pd.Series(s)
        s = df.to_numpy()
        return s

    @classmethod
    def two_stage_rnn_state(cls, j_list, r_list, curr_time, candidate_list, machine_id):
        sim = cls.get_simulator_instance()

        machine_current_time = curr_time
        if machine_id == 'None':
            previous_family_type = 0
            machine_quality = 1
        else:
            previous_family_type = r_list[machine_id].previous_family_type
            machine_quality = r_list[machine_id].machine_quality

        if Hyperparameters.maintenance == True:
            state_matrix = [[0] * 6 for _ in range(sim.number_of_job + 1)]
        else:
            state_matrix = [[0] * 6 for _ in range(sim.number_of_job)]
        for job_id in candidate_list:
            if job_id[0] == "Maintenance":
                pm_count = r_list[machine_id].maintenance_count
                pm_term = r_list[machine_id].pm_term
                row = [Hyperparameters.maintenance_time, pm_count, pm_term, machine_current_time, previous_family_type, machine_quality]
                state_matrix[-1] = row  # 마지막 행에 PM 정보 삽입
            else:
                job = j_list[job_id[0].id]
                processing_time = job.processing_times[job.oper_list[0]][machine_id]
                job_family = job.family_type
                if Hyperparameters.object == "total_tardiness":
                    due_date = job.duedate
                    row = [processing_time, due_date, job_family, machine_current_time, previous_family_type, machine_quality]
                elif Hyperparameters.object == "makespan":
                    d_rate = job.deterioration_rate
                    row = [processing_time, d_rate, job_family, machine_current_time, previous_family_type, machine_quality]
                # job_id[0].id가 전체 job 리스트에서 몇 번째인지 → 그 index에 값을 삽입
                job_str_id = job_id[0].job_id
                job_index = int(re.findall(r'\d+', job_str_id)[0])  # 숫자 부분만 추출
                state_matrix[job_index - 1] = row
        output = np.array(state_matrix, dtype=np.float32)
        return output

    @classmethod
    def js_state(cls, j_list, r_list, cur_runtime, candidate_list, machine_id):
        s = []
        number_of_jobs_wait = 0  # clear
        number_of_jobs_load = 0  # clear
        number_of_job_done = 0  # clear

        sim = cls.get_simulator_instance()
        all_p_time, remain_p_time, done_p_time = cls.all_processing_time(sim, j_list, r_list)
        machine_sums = {machine: np.sum(times) for machine, times in all_p_time.items()}
        max_machine = max(machine_sums, key=machine_sums.get)
        max_time = machine_sums[max_machine]

        wait_job_states = [0] * sim.number_of_job
        process_job_states = [0] * sim.number_of_job
        done_job_states = [0] * sim.number_of_job
        wait_family_states = [0] * Hyperparameters.family_type
        tardiness_states = [0] * sim.number_of_job
        wait_mean_processing_time = 0
        wait_due_date_state = [0] * sim.number_of_job
        for i, job_id in enumerate(j_list):  # j_list에서 job_id로 job 접근
            job = j_list[job_id]  # 각 job의 정보 접근
            job_number = int(job.job_id[1:]) - 1
            if job.status == "WAIT":
                number_of_jobs_wait += 1
                wait_mean_processing_time += sum(
                    value for times in job.processing_times.values() for value in times.values()) / len(r_list)
                wait_due_date_state[job_number] += job.duedate
                wait_job_states[job_number] += 1
                if 1 <= job.family_type <= len(wait_family_states):
                    wait_family_states[job.family_type - 1] += 1
            elif job.status == "PROCESSING":
                number_of_jobs_load += 1
                process_job_states[job_number] = 1
            elif job.status == "DONE":
                number_of_job_done += 1
                done_job_states[job_number] += 1
                tardiness = job.cal_tardiness(job.act_end_time)
                tardiness_states[job_number] += tardiness

        s.extend([w / len(j_list) for w in wait_family_states])
        s.append(number_of_jobs_wait / len(j_list))  # 1. 대기 중인 job 개수 / 총 job 개수
        s.append(number_of_jobs_load / len(j_list))  # 2. 작업 중인 job 개수 / 총 job 개수
        s.append(number_of_job_done / len(j_list))  # 6. 처리한 job 개수 / 총 job 개수

        current_time = cur_runtime
        s.append(current_time / (max_time / len(r_list)))  # 3. machine 현재 시간 / (최대 가능한 시간 / machine 개수)

        machine_setup = [0] * len(r_list)
        machine_maintenance = [0] * len(r_list)
        machine_quality = [0] * len(r_list)
        machine_deterioration_time = [0] * len(r_list)
        total_setup_number = 0
        total_maintenance_number = 0
        one_hot_family_states = []

        number_of_machines_wait = 0  # clear
        number_of_machines_pro = 0  # clear
        number_of_setup = 0
        number_of_maintenance = 0
        number_of_quality = 0
        number_of_deterioration = 0
        for i, resource in enumerate(r_list):
            machine = r_list[resource]
            if machine.status == "WAIT":
                number_of_machines_wait += 1
            elif machine.status == "PROCESSING":
                number_of_machines_pro += 1
            number_of_setup += r_list[resource].setup_count
            number_of_maintenance += r_list[resource].maintenance_count
            number_of_quality += r_list[resource].machine_quality
            number_of_deterioration += r_list[resource].deterioration_time

            # 원핫 인코딩 수정 (여러 family_type 처리)
            machine_families = r_list[resource].family_type
            if isinstance(machine_families, str):
                machine_families = [int(fam) for fam in machine_families.split(',') if
                                    fam.strip().isdigit()] if machine_families != "NONE" else []
            elif isinstance(machine_families, int):
                machine_families = [machine_families]
            one_hot = [1 if j + 1 in machine_families else 0 for j in range(Hyperparameters.family_type)]
            one_hot_family_states.extend(one_hot)

        s.append(number_of_machines_wait / len(r_list))
        s.append(number_of_machines_pro / len(r_list))
        s.append(number_of_setup / len(r_list))
        s.append(number_of_maintenance / len(r_list))
        s.append(number_of_quality / len(r_list))
        s.append(number_of_deterioration / len(r_list))

        machine_families = r_list[machine_id].family_type
        if isinstance(machine_families, str):
            machine_families = [int(fam) for fam in machine_families.split(',') if
                                fam.strip().isdigit()] if machine_families != "NONE" else []
        elif isinstance(machine_families, int):
            machine_families = [machine_families]
        one_hot = [1 if j + 1 in machine_families else 0 for j in range(Hyperparameters.family_type)]
        s.extend(one_hot)

        df = pd.Series(s)
        s = df.to_numpy()
        return s

    @classmethod
    def all_processing_time(cls, sim, j_list, r_list):
        all_p_time = {}
        remain_p_time = {}
        done_p_time = {}
        for machineId in r_list.keys():
            r_p_time = []
            a_p_time = []
            d_p_time = []
            for lotId in j_list:
                jobOperId = j_list[lotId].oper_list[0]
                # processing_time = sim.Processing_time_table[jobOperId][machineId]
                processing_time = j_list[lotId].processing_times[jobOperId][machineId]
                a_p_time.append(processing_time)
                if j_list[lotId].status == "WAIT":
                    r_p_time.append(processing_time)
                if machineId == j_list[lotId].done_machine and j_list[lotId].status == "DONE":
                    d_p_time.append(processing_time)
            all_p_time[machineId] = a_p_time
            remain_p_time[machineId] = r_p_time
            done_p_time[machineId] = d_p_time
        return all_p_time, remain_p_time, done_p_time

    @classmethod
    def get_simulator_instance(cls):
        if cls.sim is None:
            from src.simlator.Simulator import Simulator
            cls.sim = Simulator()
        return cls.sim