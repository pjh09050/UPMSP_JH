import numpy as np
from src.Object.Lot import *
from src.learner.common.Hyperparameters import *
from src.simlator.Simulator import *
import re


class StateManager:
    sim = None
    state_time = 0
    max_time_cache = None  # (lot 개수, max_time) - 에피소드 내에서 lot 구성이 같으면 재사용
    rounded_d_rate = {}  # (dataset, machine_id) -> 소수 3자리로 반올림한 deterioration_rate 배열 (데이터셋이 같으면 재사용)
    lot_arrays = None  # Simulator.lot_order 순서의 lot별 정적 정보 (family, job 번호, machine별 처리시간)

    @classmethod
    def reset_cache(cls):
        cls.max_time_cache = None
        cls.lot_arrays = None

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

        max_time = cls.get_max_time(sim, j_list, r_list)
        mean_processing_time = max_time / len(j_list)

        # WAIT lot의 family별 개수 / 처리시간 합 / 열화율 합
        # (bincount는 입력 순서대로 더하므로 lot 순서대로 더하던 기존 반복문과 결과가 같음)
        lots = cls.get_lot_arrays(sim, r_list)
        d_rate = cls.get_rounded_d_rate(sim, r_list, machine_id)
        wait = np.flatnonzero(sim.wait_mask)
        f_type = lots['family'][wait]
        in_range = (f_type >= 0) & (f_type < family_type)
        wait, f_type = wait[in_range], f_type[in_range]
        wait_family_counts = np.bincount(f_type, minlength=family_type)  # 평균 계산용 개수 저장
        wait_family_states = wait_family_counts
        wait_family_p_time = np.bincount(f_type, weights=lots['p_time'][wait, lots['machine_col'][machine_id]], minlength=family_type)  # 누적 처리시간
        wait_family_d_rate = np.bincount(f_type, weights=d_rate[lots['job_number'][wait]], minlength=family_type)  # 누적 열화율

        # family별 [대기 비율 | 평균 처리시간 | 평균 열화율]을 한 배열로 만들어 한 번에 반올림
        # (배열 np.round = 원소별 np.round, 대기 lot이 없는 family의 평균은 0.0)
        has_wait = wait_family_counts > 0
        family_features = np.concatenate([
            wait_family_states / (len(j_list) / family_type),
            np.divide(wait_family_p_time, wait_family_counts, out=np.zeros(family_type), where=has_wait) / mean_processing_time,
            np.divide(wait_family_d_rate, wait_family_counts, out=np.zeros(family_type), where=has_wait) / mean_processing_time,
        ])
        s.extend(np.round(family_features, decimals=3))

        # 현재 machine에 대한 정보
        machine_families = r_list[machine_id].family_type
        if isinstance(machine_families, str):
            machine_families = [int(fam) for fam in machine_families.split(',') if fam.strip().isdigit()] if machine_families != "NONE" else []
        elif isinstance(machine_families, int):
            machine_families = [machine_families]
        one_hot = [1 if j + 1 in machine_families else 0 for j in range(family_type)]
        s.extend(one_hot)

        machine = r_list[machine_id]
        machine_maintenance_count, machine_pm_term, machine_deterioration_time, machine_setup_count = np.round(np.array([
            machine.maintenance_count / (len(j_list)/len(r_list)),
            machine.pm_term / Hyperparameters.maintenance_time,
            machine.deterioration_time / mean_processing_time,
            machine.setup_count / len(j_list),
        ]), decimals=3)  # 서로 독립인 machine 특징은 한 번에 반올림
        # machine_maintenance_time = np.round((machine_maintenance_count * Hyperparameters.maintenance_time) / (max_time / len(j_list)), decimals=3)
        machine_quality = machine.machine_quality
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

        return np.array(s, dtype=np.float64)

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
        all_p_time = cls.all_processing_time(sim, j_list, r_list)
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
    def get_max_time(cls, sim, j_list, r_list):
        """machine별 전체 lot processing time 합의 최댓값. lot 구성이 바뀔 때만 다시 계산"""
        if cls.max_time_cache is None or cls.max_time_cache[0] != len(j_list):
            all_p_time = cls.all_processing_time(sim, j_list, r_list)
            machine_sums = {machine: np.sum(times) for machine, times in all_p_time.items()}
            max_machine = max(machine_sums, key=machine_sums.get)
            cls.max_time_cache = (len(j_list), machine_sums[max_machine])
        return cls.max_time_cache[1]

    @classmethod
    def get_lot_arrays(cls, sim, r_list):
        """Simulator.lot_order 순서의 lot별 정적 정보를 numpy 배열로 (lot 구성이 바뀔 때만 다시 만듦)
        - family: family_type - 1,  job_number: job_id 숫자 - 1
        - p_time[i, c]: lot i의 현재 operation 처리시간 (machine_col[machine_id] = c)
        """
        if cls.lot_arrays is None or cls.lot_arrays['version'] != sim.lot_order_version:
            machine_col = {machine_id: c for c, machine_id in enumerate(r_list)}
            p_time = np.zeros((len(sim.lot_order), len(machine_col)))
            for i, lot in enumerate(sim.lot_order):
                times = lot.processing_times.get(lot.current_operation_id)  # 이미 끝난 lot은 없을 수 있음 (WAIT가 아니므로 사용 안 됨)
                if times is not None:
                    p_time[i] = [times[machine_id] for machine_id in machine_col]
            cls.lot_arrays = {
                'version': sim.lot_order_version,
                'family': np.array([lot.family_type - 1 for lot in sim.lot_order], dtype=np.int64),
                'job_number': np.array([int(lot.job_id[1:]) - 1 for lot in sim.lot_order], dtype=np.int64),
                'p_time': p_time,
                'machine_col': machine_col,
            }
        return cls.lot_arrays

    @classmethod
    def get_rounded_d_rate(cls, sim, r_list, machine_id):
        """machine의 job별 열화율을 소수 3자리로 반올림한 배열 (배열 단위 np.round = 원소별 np.round 와 같은 값)"""
        key = (sim.dataSetId, machine_id)
        d_rate = cls.rounded_d_rate.get(key)
        if d_rate is None:
            d_rate = np.round(np.asarray(r_list[machine_id].deterioration_rate, dtype=np.float64), decimals=3)
            cls.rounded_d_rate[key] = d_rate
        return d_rate

    @classmethod
    def all_processing_time(cls, sim, j_list, r_list):
        """machine별로 모든 lot의 첫 operation 처리시간 리스트 {machine_id: [lot 순서의 처리시간]}"""
        all_p_time = {}
        for machineId in r_list.keys():
            all_p_time[machineId] = [j_list[lotId].processing_times[j_list[lotId].oper_list[0]][machineId] for lotId in j_list]
        return all_p_time

    @classmethod
    def get_simulator_instance(cls):
        if cls.sim is None:
            from src.simlator.Simulator import Simulator
            cls.sim = Simulator()
        return cls.sim