import random
import numpy as np
import matplotlib.pyplot as plt
from src.common import Parameters
from src.learner.common.Hyperparameters import *
from src.Object.Event import *
from collections import defaultdict
from src.Object.Resource import *

from collections import Counter
from src.learner.Manager.StateManager import *
from src.learner.Manager.RewardManager import *
from src.learner.Manager.ActionManager import *
from src.simlator.GanttChart import *
from src.master_db.DataInventory import *
from src.master_db.DB_query import *

from src.model.machine_db import *
from torch.distributions import Categorical
import torch
import pickle
import math


class Simulator:
    machine_list = {}  # id로 객체 보관
    lot_list = {}  # lot id로 객체 보관
    machine_to_factory = {}

    unload_lot_list = {}  # 아직 도착하지 못한 job, 처리가 완료된 job 기록
    load_lot_list = {}  # 작업 가능한 lot 모음

    number_of_machine = 0
    number_of_job = 0

    # todo 기본 값 세팅 해주어야함
    Q_time_table = {}  # {"j0101" : 5 , "j0203" : 5}
    Processing_time_table = {}  # {"j0101" : {"M1" : 5, "M2" : 10}, ...}
    job_info = {}  # max oper, job type, operation list

    demand_by_planhorizon = {}
    oper_in_list = {}
    rule_ratio_dict = Counter()
    rule_ratio_heatmap = {}
    actions_per_bin = []
    pm_times = []

    done = False  # 종료조건 판단
    runtime = 0  # 시간
    step_number = 0
    total_setup_time_reward = 0
    total_pm_time_reward = 0
    total_d_time_reward = 0
    total_d_p_time_reward = 0
    total_p_time_reward = 0

    cmax_time = 0
    arrival_time_index = 0
    remain_job = []

    warm_up = False
    warm_up_end = False

    event_list = []
    j = 0

    dataSetId = ''
    plan_finish = False
    sorted_event = {"warm_up_end": 3, "job_arrival": 2, "DoNothing": 2, "track_in_finish": 1, "setup_change": 1,
                    "NOTHING": 1, "Maintenance": 1}

    @classmethod
    def init_simulator(cls, dataSetId):
        try:
            cls.dataSetId = dataSetId[0]
        except:
            print("dataSetId가 존재하지 않습니다")
        DataInventory.set_db_data(cls.dataSetId)
        cls.get_job_info(cls.dataSetId)
        cls.get_machine(cls.dataSetId)

        cls.remain_job = [Hyperparameters.max_job] * Simulator.number_of_job

        cls.get_oper_info(cls.dataSetId)

        cls.predefined_due_dates = []
        cls.initial_setup_job_list = []
        cls.get_lot(cls.dataSetId)
        for _ in range(len(cls.lot_list)):
            ns = (Simulator.number_of_job + Hyperparameters.family_type) / 2
            # ns = (len(cls.lot_list) + Hyperparameters.family_type) / 2
            MP = sum(cls.all_processing_times) / len(cls.machine_list) + (ns * Hyperparameters.family_type) / len(
                cls.machine_list)
            T, R = Hyperparameters.T, Hyperparameters.R
            cls.predefined_due_dates.append(
                round(random.uniform(MP * (1 - T - R / 2) + 200, MP * (1 - T + R / 2) + 600), 2))
            # cls.predefined_due_dates.append(round(random.uniform(MP * (1 - T - R / 2), MP * (1 - T + R / 2)), 2))
            # round(random.uniform(5, 10) / 100, 2)
            cls.get_mac_status_info(cls.dataSetId)

    @classmethod
    def _init(cls, dataSetId):
        """
        machine에 먼저 접근 -> machine 객체 생성
        job에 접근 -> 비어있는 job dict만들기 , max_oper_of_job , opers_of_job
        setup에 접근 -> setup table만듬
        processing에 접근 -> processing table만듬
        oper에 접근 oper_of_job만듬
        demand에 접근 -> job 객체 생성
        """
        for dataId in dataSetId:
            cls.dataSetId = dataId
            DataInventory.set_db_data(cls.dataSetId)
            cls.get_job_info(cls.dataSetId)
            cls.get_machine(cls.dataSetId)
            cls.remain_job = [Hyperparameters.max_job] * Simulator.number_of_job
            cls.get_oper_info(cls.dataSetId)
            cls.predefined_due_dates = []
            cls.initial_setup_job_list = []
            cls.get_lot(cls.dataSetId)
            if Hyperparameters.object == "total_tardiness":
                ns = (Simulator.number_of_job + Hyperparameters.family_type) / 2
                MP = (
                                 sum(cls.all_processing_times) / Simulator.number_of_machine + ns * Hyperparameters.family_setup_time) / Simulator.number_of_machine
                T, R = Hyperparameters.T, Hyperparameters.R
                for _ in range(Simulator.number_of_job):
                    cls.predefined_due_dates.append(
                        round(random.uniform(MP * (1 - T - R / 2), MP * (1 - T + R / 2)), 3))
            cls.get_mac_status_info(cls.dataSetId)
            cls.lot_list = {}
            cls.event_list = []
            cls.machine_list = {}
            cls.family_set = {}
            cls.initial_setup_setting()
            print(f"[{cls.dataSetId}], Initial Setup List: {cls.initial_setup_job_list}")
            # cls.predefined_arrival_times = []
            # for _ in range(len(cls.lot_list) * Hyperparameters.max_job):
            #     arrival_time = random.uniform(1, cls.p_mean)
            #     cls.predefined_arrival_times.append(round(arrival_time, 2))
            # print(f"Predefined arrival times: {cls.predefined_arrival_times}")

    @classmethod
    def reset(cls, dataSetId):
        """
        하나의 episode를 시작하기 전에 시뮬레이터 상태를 초기화하고
        초기 state를 생성하여 반환하는 함수
        """
        cls.dataSetId = dataSetId
        cls.done = False
        cls.plan_finish = False
        cls.machine_list = defaultdict(Resource)
        cls.lot_list = defaultdict(Lot)
        cls.runtime = 0  # 시간
        cls.step_number = 0
        cls.total_setup_time_reward = 0
        cls.total_pm_time_reward = 0
        cls.total_d_time_reward = 0
        cls.total_d_p_time_reward = 0
        cls.total_p_time_reward = 0
        cls.cmax_time = 0
        cls.event_list = []
        cls.arrival_time_index = 0  # arrival_time 참조용 인덱스 초기화

        DataInventory.set_db_data(cls.dataSetId)
        cls.get_job_info(cls.dataSetId)
        cls.get_machine(cls.dataSetId)
        cls.family_set = {}

        cls.remain_job = [Hyperparameters.max_job] * Simulator.number_of_job
        cls.rule_ratio_dict = Counter()
        cls.rule_ratio_heatmap = {}
        cls.actions_per_bin = []
        cls.pm_times = []

        cls.get_oper_info(cls.dataSetId)
        cls.get_lot(cls.dataSetId)
        cls.get_mac_status_info(cls.dataSetId)
        if Parameters.gantt_on:
            GanttChart.reset()
        cls.initial_setup_setting()
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate(machineId, None)
        s = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job, cls.oper_in_list,
                                   candidate_list=candidate_list, candidate=None, machine_id=machineId, previous_machine_id=None)
        return s

    @classmethod
    def initial_setup_setting(cls):
        """
        데이터셋과 family 수에 따라 각 machine의 초기 setup family를 지정하는 함수
        """
        if Hyperparameters.initial_setup == True:
            if Hyperparameters.family_type == 3:
                if cls.dataSetId == "M10_J200_F3_P10-50_D5-10_Fgap_U1" or cls.dataSetId == "M10_J200_F3_P10-50_D8-13_Fgap_U1" or cls.dataSetId == "M10_J200_F3_P10-50_D8-13_Fgap_U2" or cls.dataSetId == "M10_J200_F3_P10-50_D8-13_Fgap_U3":
                    cls.initial_setup_job_list = [1, 1, 3, 3, 1, 2, 2, 3, 2, 2]
                elif cls.dataSetId == "M10_J300_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [2, 2, 3, 3, 1, 2, 1, 3, 2, 1]
                elif cls.dataSetId == "M10_J400_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [2, 1, 2, 3, 1, 2, 1, 3, 3, 1]
                elif cls.dataSetId == "M10_J500_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [1, 2, 1, 3, 3, 2, 1, 1, 2, 3]
                elif cls.dataSetId == "M25_J500_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [1, 1, 2, 2, 1, 2, 2, 3, 3, 3, 1, 3, 2, 2, 1, 2, 3, 3, 3, 3, 1, 2, 2, 3, 1]
                elif cls.dataSetId == "M25_J700_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [1, 3, 2, 3, 1, 2, 3, 1, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 3, 3, 1, 1, 3, 1, 1]
                elif cls.dataSetId == "M25_J800_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [3, 1, 2, 1, 3, 1, 2, 2, 1, 3, 2, 3, 1, 1, 2, 3, 2, 1, 3, 3, 2, 1, 3, 1, 2]
                elif cls.dataSetId == "M25_J1000_F3_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [1, 3, 2, 3, 1, 2, 1, 2, 2, 3, 2, 2, 3, 3, 3, 1, 2, 1, 1, 3, 1, 2, 1, 3, 1]
            elif Hyperparameters.family_type == 5:
                if cls.dataSetId == "M10_J200_F5_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [3, 2, 5, 3, 5, 1, 2, 3, 3, 4]
                elif cls.dataSetId == "M10_J300_F5_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [2, 1, 4, 3, 5, 5, 1, 3, 2, 1]
                elif cls.dataSetId == "M10_J400_F5_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [5, 1, 2, 3, 3, 4, 1, 5, 2, 4]
                elif cls.dataSetId == "M10_J500_F5_P10-50_D5-10_Fgap_U1" or cls.dataSetId == "M10_J500_F5_P10-50_D8-13_Fgap_U1" or cls.dataSetId == "M10_J500_F5_P10-50_D8-13_Fgap_U2" or cls.dataSetId == "M10_J500_F5_P10-50_D8-13_Fgap_U3":
                    cls.initial_setup_job_list = [3, 5, 4, 3, 5, 1, 4, 2, 2, 1]
                elif cls.dataSetId == "M25_J500_F5_P10-50_D5-10_Fgap_U1" or cls.dataSetId == "M25_J500_F5_P10-50_D8-13_Fgap_U1" or cls.dataSetId == "M25_J500_F5_P10-50_D8-13_Fgap_U2" or cls.dataSetId == "M25_J500_F5_P10-50_D8-13_Fgap_U3":
                    cls.initial_setup_job_list = [2, 4, 2, 3, 2, 1, 2, 1, 1, 1, 4, 2, 4, 1, 5, 2, 3, 5, 4, 1, 5, 4, 3, 3, 2]
                elif cls.dataSetId == "M25_J700_F5_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [4, 4, 1, 5, 4, 2, 5, 2, 4, 3, 4, 4, 5, 5, 1, 4, 1, 2, 2, 5, 1, 3, 1, 3, 3]
                elif cls.dataSetId == "M25_J800_F5_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [2, 3, 5, 4, 1, 3, 1, 2, 4, 1, 4, 2, 3, 5, 5, 2, 4, 2, 4, 3, 1, 5, 1, 3, 5]
                elif cls.dataSetId == "M25_J1000_F5_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [3, 2, 4, 5, 5, 3, 1, 2, 1, 1, 4, 2, 5, 3, 2, 1, 4, 2, 4, 3, 4, 1, 5, 3, 5]
            elif Hyperparameters.family_type == 7:
                if cls.dataSetId == "M10_J200_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [2, 1, 1, 4, 3, 6, 5, 5, 7, 4]
                elif cls.dataSetId == "M10_J300_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [2, 1, 4, 3, 5, 6, 7, 3, 2, 1]
                elif cls.dataSetId == "M10_J400_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [6, 7, 3, 2, 1, 2, 1, 4, 3, 5]
                elif cls.dataSetId == "M10_J500_F7_P10-50_D5-10_Fgap_U1" or cls.dataSetId == "M10_J500_F7_P10-50_D8-13_Fgap_U1" or cls.dataSetId == "M10_J500_F7_P10-50_D8-13_Fgap_U2" or cls.dataSetId == "M10_J500_F7_P10-50_D8-13_Fgap_U3":
                    cls.initial_setup_job_list = [3, 2, 1, 2, 1, 4, 3, 5, 6, 7]
                elif cls.dataSetId == "M25_J500_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [6, 5, 1, 7, 5, 1, 6, 3, 5, 7, 3, 3, 6, 2, 4, 5, 1, 7, 6, 6, 2, 3, 6, 5, 4]
                elif cls.dataSetId == "M25_J700_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [1, 7, 2, 1, 1, 3, 5, 6, 7, 1, 7, 1, 6, 4, 6, 5, 5, 6, 4, 1, 7, 3, 4, 3, 6]
                elif cls.dataSetId == "M25_J800_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [7, 6, 4, 6, 5, 5, 4, 4, 2, 7, 3, 4, 3, 6, 7, 2, 1, 2, 3, 5, 6, 1, 1, 2, 1]
                elif cls.dataSetId == "M25_J1000_F7_P10-50_D5-10_Fgap_U1":
                    cls.initial_setup_job_list = [7, 3, 4, 3, 6, 7, 2, 1, 2, 3, 5, 6, 1, 1, 2, 1, 6, 4, 6, 5, 5, 4, 4, 7, 7]
            for i, machine in enumerate(cls.machine_list):
                cls.machine_list[machine].family_type = cls.initial_setup_job_list[i]
                cls.machine_list[machine].previous_family_type = cls.initial_setup_job_list[i]
        else:
            for i, machine in enumerate(cls.machine_list):
                cls.machine_list[machine].family_type = 0
                cls.machine_list[machine].previous_family_type = 0

    @classmethod
    def deterioration(cls, candidate, machineId):
        job, process_time_table, setup_time, jop = candidate
        if Hyperparameters.family_setup == "family_setup":
            process_time = process_time_table
        else:
            process_time = process_time_table[machineId]
        candidate[0].previous_p_time = process_time
        candidate[0].machine_quality = cls.machine_list[machineId].machine_quality
        real_process_time = np.round(process_time / cls.machine_list[machineId].machine_quality, decimals=3)
        candidate[1] = real_process_time
        job_index = int(candidate[3][1:-2]) - 1
        deterioration_rate = np.round(cls.machine_list[machineId].deterioration_rate[job_index], decimals=3)
        candidate[0].deterioration_rate = deterioration_rate
        next_machine_quality = np.round((1 - deterioration_rate) * cls.machine_list[machineId].machine_quality, decimals=3)
        cls.machine_list[machineId].machine_quality = next_machine_quality
        return candidate

    @classmethod
    def make_job_arrival_event(cls, lot):
        """
        동적 job arrival 조건에 따라 새로운 lot을 생성하고
        job_arrival 이벤트를 event_list에 추가하는 함수
        """
        execute_condition = False  # 실행 조건 초기화
        if Hyperparameters.job_arrival_end_type == "end_time" and cls.runtime < Hyperparameters.end_time:
            execute_condition = True
        elif Hyperparameters.job_arrival_end_type == "max_job" and cls.remain_job[int(lot.job_type[1:]) - 1] > 0:
            execute_condition = True
        if execute_condition:
            job_id = lot.job_id  # candidate의 jobId를 가져옴
            demand_id = f"{job_id}_{cls.step_number}"  # 새로 생성된 demandId
            lot_id = f"{demand_id}-{job_id}"  # 새로운 Lot ID 생성
            oper_list = cls.job_info[job_id]["oper_list"]
            job_type = cls.job_info[job_id]["job_type"]
            max_oper = cls.job_info[job_id]["max_oper"]
            q_time_table = cls.get_q_time_table_of_opers(oper_list)
            # arrival time 설정
            # arrival_time = random.uniform(1, cls.p_mean)
            arrival_time = random.uniform(1, 0.2 * cls.p_mean * Simulator.number_of_job)
            new_job_arrival_time = round(cls.runtime + arrival_time, 2)
            # Due date 설정
            T, R, P = Hyperparameters.T, Hyperparameters.R, sum(cls.all_processing_times) / len(cls.machine_list) ** 2
            new_job_duedate = round(random.uniform(
                max(new_job_arrival_time + max(cls.all_processing_times) + Hyperparameters.family_setup_time,
                    P * (1 - T - R / 2)), P * (1 - T + R / 2)), 2)
            # print(f'P: {P}')
            # print(f'{max(new_job_arrival_time + max(cls.all_processing_times) + Hyperparameters.family_setup_time, P*(1-T-R/2))}~{P*(1-T+R/2)}')
            new_lot = Lot(lot_id, job_id, job_type, max_oper, new_job_duedate, new_job_arrival_time, "NOTYET",
                          oper_list, q_time_table, done_machine="None", family_type=lot.family_type)
            new_lot.processing_times = lot.processing_times.copy()
            # lot 추가 및 평균 processing time 업데이트
            cls.lot_list[lot_id] = new_lot
            cls.update_processing_time_data()
            if cls.warm_up:
                pass
            else:
                cls.remain_job[int(new_lot.job_type[1:]) - 1] -= 1
            e = Event(new_lot, "job_arrival", "NONE", new_lot.job_arrival_time, new_lot.job_arrival_time, "job_arrival",
                      "NONE", "NONE", "NONE", 0, new_lot.family_type)
            cls.event_list.append(e)
            # print(f'New Job arrival: {new_lot.current_operation_id} at time {new_job_arrival_time}, processing time: {new_lot.processing_times}, due date: {new_job_duedate}')

    @classmethod
    def process_event(cls):
        """
        event_list에서 가장 이른 종료 시점의 이벤트를 처리하고
        job과 machine 상태를 업데이트하는 이벤트 진행 함수
        """
        cls.event_list.sort(key=lambda x: (x.end_time, cls.sorted_event[x.event_type], x.machine.id if hasattr(x.machine, "id") else x.machine), reverse=False)
        if not cls.event_list:
            return

        # while cls.event_list[0].event_type == "job_arrival":
        #     earliest_time = cls.event_list[0].end_time
        #     arrival_events = [cls.event_list.pop(i) for i in reversed(range(len(cls.event_list))) if
        #                       cls.event_list[i].end_time == earliest_time and cls.event_list[
        #                           i].event_type == "job_arrival"]
        #     for event in arrival_events:
        #         event.job.arrival()
        #         # cls.make_job_arrival_event(event.job)

        earliest_time = cls.event_list[0].end_time
        # current_events = [cls.event_list.pop(i) for i in reversed(range(len(cls.event_list))) if cls.event_list[i].end_time == earliest_time]
        # 가장 이른 시각의 이벤트만 앞에서 잘라서 처리
        split_idx = 0
        n = len(cls.event_list)
        while split_idx < n and cls.event_list[split_idx].end_time == earliest_time:
            split_idx += 1
        current_events = cls.event_list[:split_idx]
        del cls.event_list[:split_idx]

        for event in current_events:
            cls.runtime = event.end_time
            if event.event_type != "track_in_finish":
                if event.event_type == "setup_change":
                    event_type = "setup"
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)
                elif event.event_type == 'Maintenance':
                    event_type = "Maintenance"
                    event.machine.wake_up_setting()
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)
                elif event.event_type == "NOTHING":
                    event_type = "NOTHING"
                    event.machine.wake_up_setting()
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)
                elif event.event_type == "warm_up_end":
                    cls.warm_up_end = True
                    continue
                elif event.event_type == 'job_arrival':
                    event.job.arrival()
                    cls.make_job_arrival_event(event.job)
                    continue
            else:
                event_type = event.job.job_type
                event.job.complete_setting(event.start_time, event.end_time, event.event_type, event.machine.id)
                event.machine.complete_setting(event.start_time, event.end_time, event.event_type)

            flow_time = event.end_time - event.start_time
            if event.job is not None:
                job_id = event.job.id
                job = event.job
                arrival_time = event.job.job_arrival_time
                due_date = event.job.duedate
            else:
                job_id = ""
                job = ""
                arrival_time = 0
                due_date = 0

            if Parameters.gantt_on or Parameters.log_hitory:
                if Hyperparameters.donothing == False and event.event_type == "NOTHING":
                    pass
                else:
                    GanttChart.save_histories(event_type, job_id, event.jop, event.start_time, event.end_time,event.machine.id, event.rule_name,
                                              event.step_num, event.q_time_diff, event.q_time_check, flow_time,event.family_type, arrival_time, due_date, job)

    @classmethod
    def get_event(cls, candidate, machineId, rule_name):
        """
        선택된 candidate에 따라 작업, setup, PM, idle 이벤트를 생성하고
        event_list에 추가하는 함수
        """
        event: Event
        step_num = cls.step_number

        if candidate == None:
            upcoming_arrival_times = [lot.job_arrival_time for lot in cls.lot_list.values() if lot.status == "NOTYET"]
            if upcoming_arrival_times:
                finish_time = min(upcoming_arrival_times)
            else:
                cls.event_list.sort(key=lambda x: [x.end_time, cls.sorted_event[x.event_type]], reverse=True)
                event = cls.event_list[0]
                finish_time = event.end_time
            if cls.runtime == finish_time:
                return
            e = Event(None, "", cls.machine_list[machineId], cls.runtime, finish_time, "NOTHING", "NONE", step_num, 0,0, "NONE")
            cls.machine_list[machineId].do_nothing_setting(finish_time)
            cls.event_list.append(e)
            cls.machine_list[machineId].nothing_action_number += 1
            cls.step_number += 1
            return

        job, process_time, setup_time, jop = candidate
        if job == 'Maintenance':
            finish_time = cls.runtime + process_time
            e = Event(None, jop, cls.machine_list[machineId], cls.runtime, finish_time, "Maintenance", "PM", step_num,
                      0, 0, "NONE")
            cls.machine_list[machineId].maintenance_count += 1
            cls.machine_list[machineId].continuous_maintenance_count += 1
            cls.event_list.append(e)
            cls.step_number += 1
            if cls.machine_list[machineId].pm_count == 0:
                cls.machine_list[machineId].pm_term = 0
            else:
                cls.machine_list[machineId].pm_term = cls.runtime - cls.machine_list[machineId].last_maintenance_time
            cls.machine_list[machineId].maintenance(finish_time)
            return
        else:
            cls.machine_list[machineId].continuous_maintenance_count = 0
            cls.machine_list[machineId].deterioration_time += process_time - job.previous_p_time
        # cls.make_job_arrival_event(job)
        # if cls.machine_list[machineId].job_id_processed == 'Maintenance':
        #     pass
        # else:
        if cls.machine_list[machineId].family_type != 0 and cls.machine_list[machineId].family_type != job.family_type:
            if Hyperparameters.family_setup == "family_setup":
                setup_time = 20
            elif Hyperparameters.family_setup == "sequence_dependent_family_setup":
                setup_time = cls.cal_setup_time(cls.machine_list[machineId], job)
            candidate[2] = setup_time
            finish_time = cls.runtime + setup_time
            e = Event(job, "setup", cls.machine_list[machineId], cls.runtime, finish_time, "setup_change", "NONE",
                      step_num, setup_time, 0, job.family_type)
            cls.machine_list[machineId].setup_count += 1
            cls.event_list.append(e)

        q_time_diff = cls.assign_setting(job, cls.machine_list[machineId], cls.runtime + setup_time + process_time)
        finish_time = np.round(cls.runtime + setup_time + process_time, 3)
        candidate.append(finish_time)
        e = Event(job, jop, cls.machine_list[machineId], cls.runtime + setup_time, finish_time, "track_in_finish",
                  rule_name, step_num, setup_time, q_time_diff, job.family_type)
        cls.machine_list[machineId].pm_term = finish_time - cls.machine_list[machineId].last_maintenance_time
        cls.event_list.append(e)
        cls.step_number += 1

    @classmethod
    def cal_setup_time(cls, machine, job):
        ## OOD 용 SETUP ####
        case_id = 2
        machine_family = machine.family_type
        job_family = job.family_type
        if machine_family == job_family:
            return 0
        # 3개 OOD test case용 seed
        seeds = [101, 202, 303]
        seed = (seeds[case_id]+ machine_family * 1000+ job_family)
        rng = np.random.default_rng(seed)
        # 100 ~ 160 inclusive
        setup_time = rng.integers(100, 161)
        return int(setup_time)
        ###### BASE SETUP ####
        # base = 70  # 25 30 50 70 70 100 100
        # mod = 61  # 17 41 51 51 61 51 101
        # A = 7  # machine_family에 곱할 계수
        # B = 11  # job_family에 곱할 계수
        # machine_family = machine.family_type
        # job_family = job.family_type
        # if machine_family == job_family:
        #     return 0
        # setup_time = base + ((machine_family * A + job_family * B) % mod)
        # return setup_time

    @classmethod
    def run(cls, rule):
        cls.initial_setup_setting()
        score = 0
        while True:
            machineId = cls.select_machine()
            candidate_list = cls.rule_get_candidate(machineId)
            candidate = Dispatcher.dispatching_rule_decision(candidate_list, rule, machineId, cls.machine_list)
            if candidate[0] == "Maintenance":
                candidate = None if candidate is None else candidate[:]
            else:
                candidate = cls.deterioration(candidate[:], machineId)
            cls.get_event(candidate, machineId, rule)
            while cls.select_machine() == "NONE":
                cls.process_event()
            if cls.rule_get_candidate(machineId) == []:
                while len(cls.event_list) != 0:
                    cls.process_event()
            if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list) and all(
                    cls.remain_job[i] == 0 for i in range(Simulator.number_of_job)):
                break
            r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list,
                                                           cls.machine_list, cls.runtime, candidate_list, candidate,
                                                           finish_job=None, episode=None)
            score += r
        Total_Flow_time, util, makespan, Total_Tardiness_time, Total_Lateness_time, T_max = cls.performance_measure()
        if Parameters.gantt_on_check:
            if Hyperparameters.object == 'total_tardiness':
                fitness = cls.performance_measure()[3]
            elif Hyperparameters.object == 'makespan':
                fitness = cls.performance_measure()[2]
            GanttChart.play_gantt(datasetId=cls.dataSetId, fitness=fitness, score=score, phase='run')
        print('dataset', Parameters.datasetId)
        print('threshold', Hyperparameters.maintenance_threshold)
        print('score:', score)
        print("Makespan:", makespan)
        return Total_Flow_time, util, makespan

    @classmethod
    def rule_select(cls, action_list, eps, phase=None):
        done = False
        phase = "second"
        finish_job = False
        while cls.select_machine() == "NONE":
            cls.process_event()
        machineId = cls.select_machine()
        if Hyperparameters.action_type == "dsp_rule_all":
            if cls.machine_list[machineId].job_id_processed == "Maintenance":
                pm_true = True
            else:
                pm_true = False
        else:
            pm_true = False
        candidate_list = cls.get_candidate(machineId, pm_true)
        rule_name, action, candidate = ActionManager.get_lot(candidate_list, action_list, cls.runtime, machineId, cls.machine_list, eps, None)
        if candidate is None or candidate[0] == "Maintenance":
            candidate = None if candidate is None else candidate[:]
        else:
            candidate = cls.deterioration(candidate[:], machineId)
        cls.get_event(candidate, machineId, rule_name)
        while cls.select_machine() == "NONE":
            cls.process_event()
        if Hyperparameters.donothing == False:
            if cls.get_candidate(machineId, pm_true) == []:
                finish_job = True
                while len(cls.event_list) != 0:
                    cls.process_event()
        if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list):
            done = True
        next_machine = cls.select_machine()
        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job, cls.oper_in_list, candidate_list, candidate, machine_id=next_machine, previous_machine_id=machineId)
        r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list, cls.machine_list, cls.runtime, candidate_list, candidate, finish_job, phase)
        cls.get_reward(candidate)
        return s_prime, action, r, done

    @classmethod
    def job_select(cls, action_list, eps, episode):
        done = False
        finish_job = False
        while cls.select_machine() == "NONE":
            cls.process_event()
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate(machineId, None)
        candidate, action, rule_name = ActionManager.get_lot(candidate_list, action_list, cls.runtime, machineId, cls.machine_list, eps, type=None)
        if candidate is None or candidate[0] == "Maintenance":
            candidate = None if candidate is None else candidate[:]
        else:
            candidate = cls.deterioration(candidate[:], machineId)
        cls.get_event(candidate, machineId, rule_name)
        while cls.select_machine() == "NONE":
            cls.process_event()
        if Hyperparameters.donothing == False:
            if cls.get_candidate(machineId, None) == []:
                finish_job = True
                while len(cls.event_list) != 0:
                    cls.process_event()
        if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list) and all(
                cls.remain_job[i] == 0 for i in range(Simulator.number_of_job)):
            done = True
        next_machine = cls.select_machine()
        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job, cls.oper_in_list, candidate_list, candidate, machine_id=next_machine, previous_machine_id=machineId)
        r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list,cls.machine_list, cls.runtime, candidate_list, candidate,finish_job, episode)
        cls.get_reward(candidate)
        return s_prime, r, done, action

    @classmethod
    def get_rule_ratio_dist(cls):
        tot = sum(cls.rule_ratio_dict.values())
        return {} if tot == 0 else {k: v / tot for k, v in cls.rule_ratio_dict.items()}

    @classmethod
    def _ensure_bin_capacity(cls, idx):
        """idx bin이 필요하면 actions_per_bin과 모든 rule 배열을 그 길이까지 0으로 확장"""
        if idx >= len(cls.actions_per_bin):
            grow = idx + 1 - len(cls.actions_per_bin)
            cls.actions_per_bin.extend([0] * grow)
            for r in cls.rule_ratio_heatmap:
                cls.rule_ratio_heatmap[r].extend([0] * grow)

    @classmethod
    def plot_rule_heatmap(cls, plot_save_directory):
        bs = 100  # bin size
        n_bins = len(cls.actions_per_bin)
        rules = sorted(cls.rule_ratio_heatmap.keys())
        if n_bins == 0 or not rules:
            print("No rule data to plot.")
            return

        # --- 분모 패딩 (보유한 bin까지만) ---
        denom_per_bin = np.zeros(n_bins, dtype=int)
        denom_per_bin[:n_bins] = np.array(cls.actions_per_bin[:n_bins], dtype=int)

        # --- 비율 행렬 (n_rules x n_bins) ---
        ratios = np.zeros((len(rules), n_bins), dtype=float)
        for j in range(n_bins):
            denom = denom_per_bin[j]
            for i, r in enumerate(rules):
                counts = cls.rule_ratio_heatmap.get(r, [])
                num = counts[j] if j < len(counts) else 0
                ratios[i, j] = (num / denom) if denom > 0 else 0.0

        # --- 축 경계 (보유한 bin까지만) ---
        draw_bins = n_bins
        x_edges = [bs * j for j in range(draw_bins)]  # 0, bs, 2bs, ...
        x_edges.append(bs * draw_bins)  # 마지막 경계 = draw_bins*bs
        y_edges = np.arange(0, len(rules) + 1, 1)

        plt.figure(figsize=(max(10, draw_bins * 0.8), max(5, len(rules) * 0.6)))
        pc = plt.pcolormesh(x_edges, y_edges, ratios, shading="flat",
                            cmap="YlGnBu", vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(pc)
        cbar.ax.tick_params(labelsize=14)

        # y 라벨은 셀 중앙에
        plt.yticks(np.arange(len(rules)) + 0.5, rules, fontsize=14)

        # x tick: 0부터 bs 단위로 draw_bins까지만
        xticks = list(range(0, bs * draw_bins + 1, bs))
        plt.xticks(xticks, fontsize=14)

        # # --- PM 시점 세로선 표시 (표시 영역 내만) ---
        # pm_times = getattr(cls, "pm_times", [])
        # for t in pm_times:
        #     if 0 <= t <= bs * draw_bins:
        #         plt.axvline(x=t, color="red", linestyle="--", linewidth=1, alpha=0.6)
        # if pm_times:
        #     # label을 안 쓰고 선만 표시하므로 legend는 생략하거나 필요시 수동 추가
        #     pass

        plt.xlabel("Time step", fontsize=14)
        plt.tight_layout()

        if Parameters.save_plot:
            result_name = (f"{Parameters.simulation_time}_{Hyperparameters.object}_"
                           f"{Hyperparameters.action_type}_{Parameters.datasetId}_"
                           f"A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_"
                           f"agent2_rule_ratio_heatmap.jpg")
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path, dpi=300)
        plt.close()

    @classmethod
    def two_step_select(cls, action_list, eps, phase):
        done = False
        finish_job = False
        while cls.select_machine() == "NONE":
            cls.process_event()
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate(machineId, phase)
        candidate, action, rule_name = ActionManager.get_lot(candidate_list, action_list, cls.runtime, machineId, cls.machine_list, eps, type=phase)
        if rule_name == "PM":
            cls.pm_times.append(int(cls.runtime))
        if phase == "second" and rule_name != "PM":
            m = re.match(r"(\d+)_([A-Za-z]+)", rule_name)
            rule = m.group(2) if m else rule_name
            cls.rule_ratio_dict[rule] += 1
            # ====== [추가] 100-step 동적 bin 집계 ======
            timestep = 100
            rt = max(1, int(cls.runtime))  # runtime=0 방어
            bin_idx = (rt - 1) // timestep  # 1~100 -> 0, 101~200 -> 1 ...
            cls._ensure_bin_capacity(bin_idx)  # 필요 시 배열 확장
            # 새 rule이면 현재 길이에 맞춰 0으로 채움
            if rule not in cls.rule_ratio_heatmap:
                cls.rule_ratio_heatmap[rule] = [0] * len(cls.actions_per_bin)
            # (안전) 혹시 rule 배열 길이가 더 짧으면 최신 길이에 맞춰 패딩
            elif len(cls.rule_ratio_heatmap[rule]) < len(cls.actions_per_bin):
                cls.rule_ratio_heatmap[rule].extend(
                    [0] * (len(cls.actions_per_bin) - len(cls.rule_ratio_heatmap[rule])))
            cls.rule_ratio_heatmap[rule][bin_idx] += 1
            cls.actions_per_bin[bin_idx] += 1
        if candidate is None or candidate[0] == "Maintenance":
            candidate = None if candidate is None else candidate[:]
        else:
            candidate = cls.deterioration(candidate[:], machineId)
        cls.get_event(candidate, machineId, rule_name)
        while cls.select_machine() == "NONE":
            cls.process_event()
        if Hyperparameters.donothing == False:
            if cls.get_candidate(machineId, None) == []:
                finish_job = True
                while len(cls.event_list) != 0:
                    cls.process_event()
        if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list) and all(cls.remain_job[i] == 0 for i in range(Simulator.number_of_job)):
            done = True
        next_machine = cls.select_machine()
        if cls.machine_list[next_machine].job_id_processed == "Maintenance":
            pm_true = 0
        else:
            pm_true = 1
        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job, cls.oper_in_list, candidate_list, candidate, machine_id=next_machine, previous_machine_id=machineId)
        r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list, cls.machine_list, cls.runtime, candidate_list, candidate, finish_job, phase)
        cls.get_reward(candidate)
        return s_prime, r, done, action, pm_true

    @classmethod
    def mc_job_select(cls, action_list, epsilon, episode):
        done = False
        finish_job = False
        while cls.select_machine() == "NONE":
            cls.process_event()
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate(machineId, None)
        candidate, action, rule_name = ActionManager.get_lot(candidate_list, action_list, cls.runtime, machineId,
                                                             cls.machine_list, epsilon, type=None)
        if candidate is None or candidate[0] == "Maintenance":
            candidate = None if candidate is None else candidate[:]
        else:
            candidate = cls.deterioration(candidate[:], machineId)
        cls.get_event(candidate, machineId, rule_name)
        while cls.select_machine() == "NONE":
            cls.process_event()
        if Hyperparameters.donothing == False:
            if cls.get_candidate(machineId, None) == []:
                finish_job = True
                while len(cls.event_list) != 0:
                    cls.process_event()
        if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list) and all(
                cls.remain_job[i] == 0 for i in range(Simulator.number_of_job)):
            done = True
        next_machine_id = cls.select_machine()
        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job, cls.oper_in_list, candidate_list, candidate, machine_id=next_machine_id)
        r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list, cls.machine_list, cls.runtime, candidate_list, candidate, finish_job, episode)
        return s_prime, r, done, action

    @classmethod
    def ppo_job_select(cls, action_list, type, eps=None):
        done = False
        finish_job = False
        while cls.select_machine() == "NONE":
            cls.process_event()
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate(machineId, None)
        candidate, action, rule_name = ActionManager.get_lot(candidate_list, action_list, cls.runtime, machineId,
                                                             cls.machine_list, eps, type)
        if candidate is None or candidate[0] == "Maintenance":
            candidate = None if candidate is None else candidate[:]
        else:
            candidate = cls.deterioration(candidate[:], machineId)
        # if candidate is None:
        #     print(f"{machineId}, nothing")
        # elif candidate[0] == "Maintenance":
        #     print(f"{machineId}, Maintenance")
        # else:
        #     print(f"{machineId}, job: {candidate[0].job_type}({candidate[0].family_type}), p_time: {candidate[1]}, d_rate: {candidate[0].deterioration_rate}")
        cls.get_event(candidate, machineId, rule_name)
        while cls.select_machine() == "NONE":
            cls.process_event()
        if Hyperparameters.donothing == False:
            if cls.get_candidate(machineId, None) == []:
                finish_job = True
                while len(cls.event_list) != 0:
                    cls.process_event()
        if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list) and all(
                cls.remain_job[i] == 0 for i in range(Simulator.number_of_job)):
            done = True
        next_machine = cls.select_machine()
        # print(next_machine)
        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,
                                         cls.oper_in_list, candidate_list, candidate, next_machine)
        # s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job, cls.oper_in_list, candidate_list, candidate, machineId)
        r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list,
                                                       cls.machine_list, cls.runtime, candidate_list, candidate,
                                                       finish_job, episode=None)
        return s_prime, r, done

    @classmethod
    def two_stage_rnn(cls, action, episode, pm_count, eps=None):
        done = False
        finish_job = False
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate_rnn(machineId, pm_count)
        candidate, action, rule_name = ActionManager.get_lot(candidate_list, action, cls.runtime, machineId,
                                                             cls.machine_list, eps, type=None)
        if candidate is None or candidate[0] == "Maintenance":
            candidate = None if candidate is None else candidate[:]
        else:
            candidate = cls.deterioration(candidate[:], machineId)
        cls.get_event(candidate, machineId, rule_name)
        while cls.select_machine() == "NONE":
            cls.process_event()
        if Hyperparameters.donothing == False:
            if cls.get_candidate_rnn(machineId, pm_count) == []:
                finish_job = True
                while len(cls.event_list) != 0:
                    cls.process_event()
        if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list):
            done = True
        next_machineId = cls.select_machine()  # next state를 구하기 위한 호출
        next_candidate_list = cls.get_candidate_rnn(machineId, pm_count)  # next state를 구하기 위한 호출
        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,
                                         cls.oper_in_list, candidate_list, next_candidate_list,
                                         machine_id=next_machineId, previous_machine_id=machineId)
        r, cls.machine_list = RewardManager.get_reward(Hyperparameters.reward_type, machineId, cls.lot_list,
                                                       cls.machine_list, cls.runtime, candidate_list, candidate,
                                                       finish_job, episode)
        if candidate[0] == "Maintenance":
            pass
        else:
            cls.machine_list[machineId].previous_family_type = candidate[0].family_type
        return s_prime, r, done, action

    @classmethod
    def get_candidate_rnn(cls, machineId, pm_count):
        candidate_list = []
        for lot in cls.lot_list.values():
            setup_time = 0
            if lot.status == "WAIT" and lot.job_arrival_time <= cls.runtime:
                for operation, machine_times in lot.processing_times.items():
                    if Hyperparameters.family_setup == "family_setup":
                        processing_time = machine_times.get(machineId)
                    else:
                        processing_time = machine_times
                    if processing_time is not None:
                        candidate_list.append([lot, processing_time, setup_time, operation])
                        break
        # 유지보수 candidate 넣기
        if Hyperparameters.maintenance == True and pm_count == False:
            if all(cls.lot_list[lot].status != "WAIT" for lot in cls.lot_list):
                pass
            else:
                candidate_list.append(['Maintenance', Hyperparameters.maintenance_time, 0, 'Maintenance'])
        return candidate_list

    @classmethod
    def available_jobs_mask(cls, pm_count):
        machineId = cls.select_machine()
        if Hyperparameters.action_type == 'two_stage_rnn':
            candidate_list = cls.get_candidate_rnn(machineId, pm_count)
        else:
            candidate_list = cls.get_candidate(machineId, None)
        if Hyperparameters.maintenance == True:
            candidate_to_setup_list = [candidate[0].job_type if hasattr(candidate[0], 'job_type') else candidate[3] for
                                       candidate in candidate_list]
        else:
            candidate_to_setup_list = [candidate[0].job_type for candidate in candidate_list]  # setup list
        setup_index = [1 if setup in candidate_to_setup_list else 0 for action, setup in
                       enumerate(Hyperparameters.action_dimension)]
        if pm_count == True:
            setup_index[-1] = 0
        return setup_index

    @classmethod
    def available_family_mask(cls):
        machineId = cls.select_machine()
        candidate_list = cls.get_candidate(machineId, None)
        if Hyperparameters.maintenance == True:  # PM 추가해줘야함
            candidate_to_family_type = sorted(
                set(candidate[0].family_type for candidate in candidate_list if candidate[0] != 'Maintenance'))
        else:
            candidate_to_family_type = sorted(set(candidate[0].family_type for candidate in candidate_list))
        family_mask = [int(family in candidate_to_family_type) for family in range(1, Hyperparameters.family_type + 1)]
        if Hyperparameters.maintenance:
            pm_flag = 1 if sum(family_mask) > 0 else 0
            family_mask.append(pm_flag)
        return family_mask

    @classmethod
    def get_valid_actions(cls, s_prime):
        if (Hyperparameters.action_type == "family_type" or Hyperparameters.action_type == "ppo_family_type" or Hyperparameters.action_type == "family_per_machine"):
            if Hyperparameters.state_type == "norm_state":
                wait_family = s_prime[:, Simulator.number_of_job:Simulator.number_of_job + Hyperparameters.family_type]
            else:
                wait_family = s_prime[:, :Hyperparameters.family_type]
            valid_actions = (wait_family != 0)
        else:
            wait_jobs = s_prime[:, :Simulator.number_of_job]
            valid_actions = (wait_jobs == 1)
        if Hyperparameters.maintenance:
            # Step 4: PM 추가 여부 결정 (하나라도 True이면 True, 모두 False이면 False)
            pm_action = valid_actions.any(dim=1, keepdim=True)
            valid_actions = torch.cat([valid_actions, pm_action], dim=1)
        return valid_actions

    @classmethod
    def get_valid_two_step_actions(cls, s_prime):
        wait_family = s_prime[:, :Hyperparameters.family_type]
        valid_actions = (wait_family != 0)
        if Hyperparameters.action_type == "two_stage_dqn_rule":
            valid_actions = valid_actions.repeat_interleave(Hyperparameters.rule_count, dim=1)
        elif Hyperparameters.action_type == "dsp_rule":
            valid_actions = valid_actions.repeat_interleave(Hyperparameters.rule_count, dim=1)
            pm_term = s_prime[:, -5:-4]
            pm_count = s_prime[:, -6:-5]
            if (pm_term != 0).any() and (pm_count != 0).any():
                extra = torch.ones((s_prime.size(0), 1), dtype=torch.bool, device=s_prime.device)  # PM 추가
            else:
                extra = torch.zeros((s_prime.size(0), 1), dtype=torch.bool, device=s_prime.device)
            valid_actions = torch.cat([valid_actions, extra], dim=1)
        return valid_actions

    @classmethod
    def get_reward(cls, candidate):
        r = 0
        if candidate[0] == "Maintenance":
            pm_time = Hyperparameters.maintenance_time
            r -= pm_time
            cls.total_pm_time_reward -= pm_time
        else:
            deterioration_time = candidate[1] - candidate[0].previous_p_time
            setup_time = candidate[2]
            r -= setup_time + deterioration_time
            cls.total_setup_time_reward -= setup_time
            cls.total_d_time_reward -= deterioration_time
            cls.total_d_p_time_reward -= candidate[1]
            cls.total_p_time_reward -= candidate[0].previous_p_time

    @classmethod
    def write_reward_txt(cls, makespan, reward, save_directory):
        text_line = (
            f"makespan : {makespan} \n"
            f"누적 reward : {reward} \n"
            f"pm time : {cls.total_pm_time_reward} \n"
            f"setup time : {cls.total_setup_time_reward} \n"
            f"deterioration time : {cls.total_d_time_reward} \n"
            f"previous p time : {cls.total_p_time_reward} \n"
            f"p time : {cls.total_d_p_time_reward} \n"
        )
        if Parameters.save_plot:
            txt_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Hyperparameters.action_type}_{Parameters.datasetId}_A{Hyperparameters.rule_count}_F{Hyperparameters.family_type}_reward_result.txt"
            txt_path = os.path.join(save_directory, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_line)
            print(f"reward 저장 완료: {txt_path}")

    @classmethod
    def gantt_chart(cls, score, phase):
        if Parameters.gantt_on:
            if Hyperparameters.object == 'total_tardiness':
                fitness = cls.performance_measure()[3]
            elif Hyperparameters.object == 'makespan':
                fitness = cls.performance_measure()[2]
            GanttChart.play_gantt(cls.dataSetId, fitness, score, phase)

    @classmethod
    def assign_setting(cls, job, machine, reservation_time):  # job = 1 machine = 1
        q_time_diff = job.assign_setting(machine, reservation_time)
        machine.assign_setting(job, reservation_time)
        return q_time_diff

    @classmethod
    def select_machine(cls):
        selected_machine = "NONE"
        for machineId in cls.machine_list:
            if cls.machine_list[machineId].status == "WAIT":
                selected_machine = machineId
                break
        return selected_machine

    @classmethod
    def get_candidate(cls, machineId, phase):
        candidate_list = []
        for lot in cls.lot_list.values():
            setup_time = 0
            if lot.status == "WAIT" and lot.job_arrival_time <= cls.runtime:
                for operation, machine_times in lot.processing_times.items():
                    if Hyperparameters.family_setup == "family_setup":
                        processing_time = machine_times.get(machineId)
                    else:
                        processing_time = machine_times
                    if processing_time is not None:
                        candidate_list.append([lot, processing_time, setup_time, operation])
                        break
        # 유지보수 candidate 넣기
        if Hyperparameters.action_type == "dsp_rule":
            if phase == True:
                pass
            else:
                if candidate_list == []:
                    pass
                else:
                    candidate_list.append(['Maintenance', Hyperparameters.maintenance_time, 0, 'Maintenance'])
        else:
            if Hyperparameters.maintenance == True and phase != "second":
                if all(cls.lot_list[lot].status != "WAIT" for lot in cls.lot_list):
                    pass
                else:
                    candidate_list.append(['Maintenance', Hyperparameters.maintenance_time, 0, 'Maintenance'])
        return candidate_list

    @classmethod
    def rule_get_candidate(cls, machineId):
        candidate_list = []
        for lot in cls.lot_list.values():
            setup_time = 0
            if lot.status == "WAIT" and lot.job_arrival_time <= cls.runtime:
                for operation, machine_times in lot.processing_times.items():
                    if Hyperparameters.family_setup == "family_setup":
                        processing_time = machine_times.get(machineId)
                    else:
                        processing_time = machine_times
                    if processing_time is not None:
                        candidate_list.append([lot, processing_time, setup_time, operation])
                        break
        # 디스패칭룰 액션에서 임계값 아래일때 maintenance 강제로 진행
        if Hyperparameters.maintenance == True:
            if all(cls.lot_list[lot].status != "WAIT" for lot in cls.lot_list):
                pass
            else:
                if cls.machine_list[machineId].machine_quality <= Hyperparameters.maintenance_threshold:
                    candidate_list.append(['Maintenance', Hyperparameters.maintenance_time, 0, 'Maintenance'])
        return candidate_list

    @classmethod
    def get_machine(cls, dataSetId):
        # todo 해당 데이터 셋에 해당하는 기계정보를 전부 가져옴 -> 기계 id를
        # todo 기계 정보를 이용해 machine 객체들을 생성함
        # 생성한 객체들을 machine_list에 딕셔너리 형태로 저장함
        machines = DataInventory.get_machine_db_data()
        machines = sorted(machines, key=lambda x: int(x.machineId[1:]))  # Assuming machineId starts with 'M'
        cls.number_of_machine = len(machines)
        for machine in machines:
            machine_quality = 1
            last_time = 0
            # setup_time_table = cls.get_setup_time_table(dataSetId, machine)
            setup_time_table = []
            r = Resource(machine.machineId, machine.machineType, setup_time_table, machine.factoryId,
                         machine.deterioration_rate, machine_quality, last_time, "NONE", "NONE")
            cls.machine_list[machine.machineId] = r
            cls.machine_to_factory[machine.machineId] = machine.factoryId

    @classmethod
    def get_lot(cls, dataSetId):
        jobs = DataInventory.get_demand_db_data()
        cls.lot_list = {}
        cls.family_set = {}

        job_processing_times = {}
        all_processing_times = []
        for i, job in enumerate(jobs):
            job_processing_times[job.jobId] = {}
            oper_list = cls.job_info[job.jobId]["oper_list"]
            for oper in oper_list:
                job_processing_times[job.jobId][oper] = {}
                for machineId in cls.machine_list:
                    processing_time = cls.Processing_time_table.get(oper, {}).get(machineId, 0)
                    job_processing_times[job.jobId][oper][machineId] = processing_time
                    all_processing_times.append(processing_time)

        cls.p_mean = sum(all_processing_times) / len(all_processing_times)
        cls.all_processing_times = all_processing_times

        # 각 family_type에 균등하게 job을 분배
        family_num = Hyperparameters.family_type
        family_types = list(range(1, family_num + 1))  # family_type 리스트 생성
        family_type_allocation = []
        for i in range(len(jobs)):
            family_type_allocation.append(family_types[i % family_num])

        # family_set key 미리 생성
        for family_type in family_types:
            cls.family_set[family_type] = {}

        # job 생성
        a, d = 0, 0
        for idx, job in enumerate(jobs):
            for instance_id in range(Hyperparameters.max_job):
                lot_id = f"{job.demandId}-{job.jobId}-{instance_id}"
                status = "WAIT"
                oper_list = cls.job_info[job.jobId]["oper_list"]
                q_time_table = {}
                family_type = family_type_allocation[idx]
                # # due date 설정
                # P = sum(all_processing_times) / len(cls.machine_list)
                # T, R = Hyperparameters.T, Hyperparameters.R
                # job.duedate = round(random.uniform(P*(1-T-R/2), P*(1-T+R/2)), 2)
                j = Lot(lot_id, job.jobId, cls.job_info[job.jobId]["job_type"], cls.job_info[job.jobId]["max_oper"],
                        job.duedate, 0, status, oper_list, q_time_table, done_machine='None', family_type=family_type)
                for oper in oper_list:
                    j.processing_times[oper] = {machineId: job_processing_times[job.jobId][oper][machineId] for machineId in cls.machine_list}
                if cls.predefined_due_dates:
                    j.duedate = cls.predefined_due_dates[a]
                    a += 1
                cls.lot_list[lot_id] = j
                # family_set에도 같이 저장
                cls.family_set[family_type][lot_id] = j

                cls.remain_job[int(j.job_type[1:]) - 1] -= 1
            d += 1

    @classmethod
    def get_mac_status_info(cls, dataSetId):
        mac_status = DataInventory.get_mac_status_db_data()
        if len(mac_status) == 0:
            return
        for mac_row in mac_status:
            machine = cls.machine_list[mac_row.machineId]  # machine 객체
            # 새로운 Lot 생성
            job_id = mac_row.jobId
            j = Lot(mac_row.lotId, mac_row.jobId, cls.job_info[job_id]['job_type'], cls.job_info[job_id]["max_oper"]
                    , mac_row.dueDate, 0, "WAIT", cls.job_info[job_id]["oper_list"],
                    cls.get_q_time_table_of_opers(cls.job_info[mac_row.jobId]["oper_list"]))
            cls.lot_list[mac_row.lotId] = j

            job = cls.lot_list[mac_row.lotId]
            _ = cls.assign_setting(job, machine, mac_row.finishTime)
            step_num = cls.step_number
            cls.step_number += 1
            e = Event(job, job.current_operation_id, machine, 0, mac_row.finishTime,
                      "track_in_finish", None, step_num, 0, 0, j.family_type)
            cls.event_list.append(e)

    @classmethod
    def get_job_info(cls, dataSetId):
        jobs = DataInventory.get_job_db_data()
        cls.number_of_job = len(jobs)
        # print(type(jobs[0]))
        for job in jobs:
            job_info = {}
            job_info["max_oper"] = job.maxOper
            job_info["job_type"] = job.jobType
            # oper_list = DB_query.get_all_operation_of_job(dataSetId,Oper_db,job.jobId)
            oper_list = DataInventory.sim_data.get_oper_list_by_job(job.jobId)
            job_info["oper_list"] = oper_list
            cls.job_info[job.jobId] = job_info

    @classmethod
    def get_oper_info(cls, dataSetId):
        opers = DataInventory.get_oper_db_data()
        for oper in opers:
            cls.Q_time_table[oper.operId] = oper.operQtime
            for machineId in cls.machine_list:
                """processing_time = DB_query.get_processing_time(dataSetId, ProcessingTime_db,oper.operId,
                                                               cls.machine_list[machineId].machine_type)"""
                processing_time = DataInventory.sim_data.get_processing_time_by_oper_and_machine(oper.operId,
                                                                                                 cls.machine_list[
                                                                                                     machineId].machine_type)
                if oper.operId not in cls.Processing_time_table:
                    cls.Processing_time_table[oper.operId] = {}
                cls.Processing_time_table[oper.operId][machineId] = processing_time

    @classmethod
    def get_setup_time_table(cls, dataSetId, machine):
        from_to_setup_time_dict = DataInventory.sim_data.get_setup_time_list_by_machine(machine.machineId)
        return from_to_setup_time_dict

    @classmethod
    def get_q_time_table_of_opers(cls, oper_list):  # 해당 job의
        q_time_table = {}
        for oper in oper_list:
            q_time_table[oper] = cls.Q_time_table[oper]
        return q_time_table

    @classmethod
    def can_process_oper_in_machine(cls, job, machineId):
        if cls.Processing_time_table[job.current_operation_id][machineId] == 0:
            return False
        else:
            return True

    @classmethod
    def performance_measure(cls):  # 성능 지표 도출에 사용됨
        Total_Flow_time = 0
        Total_Tardiness_time = 0  # new
        Total_Lateness_time = 0  # new
        T_max = 0  # new
        value_time_table = []
        full_time_table = []

        for machineId in cls.machine_list:
            value_added_time, full_time = cls.machine_list[machineId].cal_util()
            value_time_table.append(value_added_time)
            full_time_table.append(full_time)

        util = sum(value_time_table) / sum(full_time_table)
        makespan = max([machine.last_work_finish_time for machine in cls.machine_list.values()])

        filtered_lot_list = {lotId: lot for lotId, lot in cls.lot_list.items()}
        for lotId, lot in filtered_lot_list.items():
            Total_Flow_time += lot.job_flowtime
            Total_Tardiness_time += lot.tardiness_time
            Total_Lateness_time += lot.lateness_time
            if lot.tardiness_time > T_max:
                T_max = lot.tardiness_time
        return Total_Flow_time, util, np.round(makespan, decimals=3), np.round(Total_Tardiness_time,
                                                                               decimals=3), Total_Lateness_time, T_max

    @classmethod
    def update_processing_time_data(cls):
        all_processing_times = []
        for lot in cls.lot_list.values():
            for machineId in cls.machine_list:
                job_id = list(lot.processing_times.keys())[0]
                processing_time = lot.processing_times[job_id][machineId]
                all_processing_times.append(processing_time)
        if all_processing_times:
            cls.p_mean = sum(all_processing_times) / len(all_processing_times)
            cls.all_processing_times = all_processing_times
        else:
            cls.p_mean = 0

    @classmethod
    def get_fittness_with_meta_heuristic(cls, job_seq, mac_seq, pm_seq, gantt_on):
        # chromosome = [[machine seq], [job seq], [pm seq]]
        for i in range(len(job_seq)):
            lotId = job_seq[i]
            machineId = mac_seq[i]
            pm_true = pm_seq[i]
            jobOperId = cls.lot_list[lotId].current_operation_id
            candidate = ([cls.lot_list[lotId], cls.Processing_time_table[jobOperId], 0, jobOperId])
            candidate = cls.deterioration(candidate[:], machineId)
            cls.get_event_meta(candidate, machineId)
            while cls.event_list:
                cls.process_event_meta()
            if pm_true == 1:
                candidate = ['Maintenance', Hyperparameters.maintenance_time, 0, 'Maintenance']
                cls.get_event_meta(candidate, machineId)
                # print(f"{machineId}, Maintenance")
            while cls.event_list:
                cls.process_event_meta()

        makespan = 0
        for machine in cls.machine_list:
            if makespan < cls.machine_list[machine].last_work_finish_time:
                makespan = cls.machine_list[machine].last_work_finish_time
        if gantt_on == True:
            cls.gantt_chart(makespan, 'GA')
        else:
            cls.reset(cls.dataSetId)
        return makespan

    @classmethod
    def get_event_meta(cls, candidate, machineId):
        step_num = cls.step_number
        job, process_time, setup_time, jop = candidate
        if job == "Maintenance":
            start_time = cls.machine_list[machineId].last_work_finish_time
        else:
            start_time = max(cls.machine_list[machineId].last_work_finish_time, job.act_end_time)
        if job == 'Maintenance':
            finish_time = start_time + process_time
            e = Event(None, jop, cls.machine_list[machineId], start_time, finish_time, "Maintenance", "PM", step_num, 0,
                      0, "NONE")
            cls.machine_list[machineId].maintenance_count += 1
            cls.machine_list[machineId].continuous_maintenance_count += 1
            cls.event_list.append(e)
            cls.machine_list[machineId].maintenance(finish_time)
            cls.step_number += 1
            cls.machine_list[machineId].pm_term = finish_time - cls.machine_list[machineId].last_maintenance_time
            return
        else:
            cls.machine_list[machineId].continuous_maintenance_count = 0
            cls.machine_list[machineId].deterioration_time += process_time - job.previous_p_time

        if cls.machine_list[machineId].family_type != 0 and cls.machine_list[machineId].family_type != job.family_type:
            if Hyperparameters.family_setup == "family_setup":
                setup_time = 20
            elif Hyperparameters.family_setup == "sequence_dependent_family_setup":
                setup_time = cls.cal_setup_time(cls.machine_list[machineId], job)
            candidate[2] = setup_time
            finish_time = start_time + setup_time
            e = Event(job, "setup", cls.machine_list[machineId], start_time, finish_time, "setup_change", "NONE",
                      step_num, setup_time, 0, job.family_type)
            cls.machine_list[machineId].setup_count += 1
            cls.event_list.append(e)

        q_time_diff = cls.assign_setting(job, cls.machine_list[machineId], start_time + setup_time + process_time)
        finish_time = np.round(start_time + setup_time + process_time, 3)
        candidate.append(finish_time)
        e = Event(job, jop, cls.machine_list[machineId], start_time + setup_time, finish_time, "track_in_finish",
                  "meta", step_num, setup_time, q_time_diff, job.family_type)
        cls.machine_list[machineId].pm_term = finish_time - cls.machine_list[machineId].last_maintenance_time
        cls.event_list.append(e)
        cls.step_number += 1

    @classmethod
    def process_event_meta(cls):
        cls.event_list.sort(key=lambda x: (x.end_time, cls.sorted_event[x.event_type],
                                           x.machine.id if hasattr(x.machine, "id") else x.machine), reverse=False)
        earliest_time = cls.event_list[0].end_time
        current_events = [cls.event_list.pop(i) for i in reversed(range(len(cls.event_list))) if
                          cls.event_list[i].end_time == earliest_time]
        for event in current_events:
            cls.runtime = event.end_time
            if event.event_type != "track_in_finish":
                if event.event_type == "setup_change":
                    # print(f'event 진행: {event.event_type}, 실행 machine: {event.machine.id}, job type: {event.job.job_type}')
                    event_type = "setup"
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)  # 기계도 사용가능하도록 변함
                elif event.event_type == 'Maintenance':
                    # print(f'event 진행: {event.event_type}, 실행 machine: {event.machine.id}')
                    event_type = "Maintenance"
                    event.machine.wake_up_setting()
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)  # 기계도 사용가능하도록 변함
                elif event.event_type == "NOTHING":
                    # print(f'event 진행: {event.event_type}, 실행 machine: {event.machine.id}')
                    event_type = "NOTHING"
                    event.machine.wake_up_setting()
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)  # 기계도 사용가능하도록 변함
                elif event.event_type == "warm_up_end":
                    cls.warm_up_end = True
                    continue
                elif event.event_type == 'job_arrival':
                    # print(f'event 진행: {event.event_type}, job도착: {event.job.job_id}')
                    event.job.arrival()
                    cls.make_job_arrival_event(event.job)
                    continue
            else:
                # print(f'event 진행: {event.event_type}, 실행 machine: {event.machine.id}, job type: {event.job.job_type}')
                event_type = event.job.job_type
                event.job.complete_setting(event.start_time, event.end_time, event.event_type,
                                           event.machine.id)  # 작업이 대기로 변함, 시작시간, 종료시간, event_type
                event.machine.complete_setting(event.start_time, event.end_time, event.event_type)  # 기계도 사용가능하도록 변함
            flow_time = event.end_time - event.start_time
            if event.job is not None:
                job_id = event.job.id
                job = event.job
                arrival_time = event.job.job_arrival_time
                due_date = event.job.duedate
            else:
                job_id = ""
                job = ""
                arrival_time = 0
                due_date = 0
            if Parameters.gantt_on or Parameters.log_hitory:
                GanttChart.save_histories(event_type, job_id, event.jop, event.start_time, event.end_time,
                                          event.machine.id, event.rule_name, event.step_num, event.q_time_diff,
                                          event.q_time_check, flow_time, event.family_type, arrival_time, due_date, job)

    @classmethod
    def get_job_seq(cls):
        job_seq = []
        for i in cls.lot_list:
            job_seq.append(i)
        return job_seq

    @classmethod
    def get_random_machine(cls, job):
        operId = cls.lot_list[job].current_operation_id
        mac_list = cls.Processing_time_table[operId]
        change_mac_list = []
        for mac, p_time in mac_list.items():
            if p_time != 0:
                change_mac_list.append(mac)
        return random.choice(change_mac_list)