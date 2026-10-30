from collections import Counter, defaultdict
from src.common import Parameters
from src.learner.common.Hyperparameters import *
import re
import numpy as np

class Dispatcher:
    @classmethod
    def dispatching_rule_decision(cls, candidate_list, rule_name, machineId, machine_list, candidate_by_family=None):
        if Hyperparameters.action_type == "two_stage_dqn_rule":
            match = re.match(r"(\d+)_([A-Za-z]+)", rule_name)
            if match:
                family_type = match.group(1)
                rule = match.group(2)
                candidate = cls.two_step_dqn_rule(family_type, rule, candidate_by_family, machine_list, machineId)
                return candidate
        elif Hyperparameters.action_type == "dsp_rule":
            match = re.match(r"(\d+)_([A-Za-z]+)", rule_name)
            if match:
                family_type = match.group(1)
                rule = match.group(2)
            else:
                family_type = None
                rule = "PM"
            candidate = cls.dsp_rule(family_type, rule, candidate_list, machine_list, machineId)
            return candidate
        elif rule_name == "SPT":
            candidate_list = cls.dispatching_rule_SPT(candidate_list)
        elif rule_name == "LPT":
            candidate_list = cls.dispatching_rule_LPT(candidate_list)
        elif rule_name == "SSU":
            candidate_list = cls.dispatching_rule_SSU(candidate_list)
        elif rule_name == "MOR":
            candidate_list = cls.dispatching_rule_MOR(candidate_list)
        elif rule_name == "LOR":
            candidate_list = cls.dispatching_rule_LOR(candidate_list)
        elif rule_name == "EDD":
            candidate_list = cls.dispatching_rule_EDD(candidate_list)
        elif rule_name == "FIFO":
            candidate_list = cls.dispatching_rule_FIFO(candidate_list)
        elif rule_name == "LIFO":
            candidate_list = cls.dispatching_rule_LIFO(candidate_list)
        elif rule_name == "SETUP_LPT":
            candidate_list = cls.dispatching_rule_SETUP_LPT(candidate_list, machine_list)
        elif rule_name == "Family_LPT":
            candidate_list = cls.dispatching_rule_Family_LPT(candidate_list, machineId)
        elif rule_name == "PM":
            candidate_list = cls.dispatching_rule_PM(candidate_list)
        elif rule_name == "SDR":
            candidate_list = cls.dispatching_rule_SDR(candidate_list, machine_list, machineId)
        elif rule_name == "LDR":
            candidate_list = cls.dispatching_rule_LDR(candidate_list, machine_list, machineId)
        elif rule_name == "Family_LDR":
            candidate_list = cls.dispatching_rule_Family_LDR(candidate_list, machine_list, machineId)
        return candidate_list

    @classmethod
    def dsp_rule(cls, family_type, rule, candidate_list, machine_list, machineId):
        if rule == "PM":
            candidate = next(filter(lambda item: item[0] == "Maintenance", candidate_list), None)
            return candidate
        else:
            filtered_candidates = [c for c in candidate_list if c[0] != "Maintenance" and c[0].family_type == int(family_type)]
            if rule == "SPTSDR":
                best_candidate = None
                min_cal_p = float('inf')
                for candidate in filtered_candidates:
                    job_index = int(candidate[3][1:-2]) - 1
                    deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
                    if Hyperparameters.family_type == "family_setup":
                        p_curr = candidate[1]  # per-candidate p
                        cal_p = (p_curr * deterioration_rate) / (1 - deterioration_rate)
                    else:
                        p_curr = candidate[1][machineId]  # per-machine p
                        cal_p = (p_curr * deterioration_rate) / (1 - deterioration_rate)
                    if best_candidate is None:
                        min_cal_p = cal_p
                        best_candidate = candidate
                    else:
                        p_best = best_candidate[1] if Hyperparameters.family_type == "family_setup" else \
                        best_candidate[1][machineId]
                        best_job_index = int(best_candidate[3][1:-2]) - 1
                        d_best = machine_list[machineId].deterioration_rate[best_job_index]
                        if (cal_p < min_cal_p or (cal_p == min_cal_p and (p_curr < p_best or (p_curr == p_best and deterioration_rate < d_best)))):
                            min_cal_p = cal_p
                            best_candidate = candidate
                return best_candidate
            elif rule == "SPTLDR":
                best_candidate = None
                min_cal_p = float('inf')
                for candidate in filtered_candidates:
                    job_index = int(candidate[3][1:-2]) - 1
                    deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
                    if Hyperparameters.family_type == "family_setup":
                        p_curr = candidate[1]
                        cal_p = p_curr / (deterioration_rate * (1 - deterioration_rate))
                    else:
                        p_curr = candidate[1][machineId]
                        cal_p = p_curr / (deterioration_rate * (1 - deterioration_rate))
                    if best_candidate is None:
                        min_cal_p = cal_p
                        best_candidate = candidate
                    else:
                        best_job_index = int(best_candidate[3][1:-2]) - 1
                        p_best = best_candidate[1] if Hyperparameters.family_type == "family_setup" else \
                        best_candidate[1][machineId]
                        d_best = machine_list[machineId].deterioration_rate[best_job_index]
                        if (cal_p < min_cal_p or (cal_p == min_cal_p and (p_curr < p_best or (p_curr == p_best and deterioration_rate < d_best)))):
                            min_cal_p = cal_p
                            best_candidate = candidate
                return best_candidate
            elif rule == "LPTSDR":
                best_candidate = None
                max_cal_p = float('-inf')
                for candidate in filtered_candidates:
                    job_index = int(candidate[3][1:-2]) - 1
                    deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
                    if Hyperparameters.family_type == "family_setup":
                        p_curr = candidate[1]  # per-candidate p
                        cal_p = (p_curr * deterioration_rate) / (1 - deterioration_rate)
                    else:
                        p_curr = candidate[1][machineId]  # per-machine p
                        cal_p = (p_curr * deterioration_rate) / (1 - deterioration_rate)
                    if best_candidate is None:
                        max_cal_p = cal_p
                        best_candidate = candidate
                    else:
                        p_best = best_candidate[1] if Hyperparameters.family_type == "family_setup" else \
                        best_candidate[1][machineId]
                        best_job_index = int(best_candidate[3][1:-2]) - 1
                        d_best = machine_list[machineId].deterioration_rate[best_job_index]
                        if (cal_p > max_cal_p or (cal_p == max_cal_p and (p_curr > p_best or (p_curr == p_best and deterioration_rate < d_best)))):
                            max_cal_p = cal_p
                            best_candidate = candidate
                return best_candidate
            elif rule == "LPTLDR":
                best_candidate = None
                max_cal_p = float('-inf')
                for candidate in filtered_candidates:
                    job_index = int(candidate[3][1:-2]) - 1
                    deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
                    machine_quality = machine_list[machineId].machine_quality
                    if Hyperparameters.family_type == "family_setup":
                        p_curr = candidate[1]
                        cal_p = p_curr / (deterioration_rate * (1 - deterioration_rate))
                    else:
                        p_curr = candidate[1][machineId]
                        cal_p = p_curr / (deterioration_rate * (1 - deterioration_rate))
                        # cal_p = (p_curr * (1 - deterioration_rate)) / deterioration_rate
                    if best_candidate is None:
                        max_cal_p = cal_p
                        best_candidate = candidate
                    else:
                        best_job_index = int(best_candidate[3][1:-2]) - 1
                        p_best = best_candidate[1] if Hyperparameters.family_type == "family_setup" else \
                        best_candidate[1][machineId]
                        d_best = machine_list[machineId].deterioration_rate[best_job_index]
                        if (cal_p > max_cal_p or (cal_p == max_cal_p and (p_curr > p_best or (p_curr == p_best and deterioration_rate < d_best)))):
                            max_cal_p = cal_p
                            best_candidate = candidate
                return best_candidate
            elif rule == "SPT":
                if Hyperparameters.family_type == "family_setup":
                    candidate = min(filtered_candidates, key=lambda x: x[1])
                else:
                    candidate = min(filtered_candidates, key=lambda x: x[1][machineId])
            elif rule == "LPT":
                if Hyperparameters.family_type == "family_setup":
                    candidate = max(filtered_candidates, key=lambda x: x[1])
                else:
                    candidate = max(filtered_candidates, key=lambda x: x[1][machineId])
            elif rule == "SDR":
                if Hyperparameters.family_type == "family_setup":
                    candidate = min(filtered_candidates, key=lambda x: x[0].deterioration_rate)
                else:
                    candidate = min(filtered_candidates,
                                    key=lambda x: machine_list[machineId].deterioration_rate[int(x[3][1:-2]) - 1])
            elif rule == "LDR":
                if Hyperparameters.family_type == "family_setup":
                    candidate = max(filtered_candidates, key=lambda x: x[0].deterioration_rate)
                else:
                    candidate = max(filtered_candidates,
                                    key=lambda x: machine_list[machineId].deterioration_rate[int(x[3][1:-2]) - 1])
            return candidate

    @classmethod
    def two_step_dqn_rule(cls, family_type, rule, candidate_by_family, machine_list, machineId):
        """선택된 family 안에서 rule에 따라 candidate 하나를 고름 (후보가 없으면 None)
        - SPTSDR/SPTLDR/LPTSDR/LPTLDR: 처리시간(p)과 열화율(d)을 합친 점수 cal_p 기준
            SDR: cal_p = p*d/(1-d),  LDR: cal_p = p/(d*(1-d))
            SPT: cal_p 최소 -> p 최소 -> d 최소,  LPT: cal_p 최대 -> p 최대 -> d 최소 (완전 동점이면 앞선 후보)
        - SPT/LPT: p 최소/최대,  SDR/LDR: d 최소/최대
        """
        filtered_candidates = candidate_by_family.get(int(family_type), [])
        if not filtered_candidates:
            return None
        deterioration_rate = machine_list[machineId].deterioration_rate

        def p_time(candidate):  # 이 machine에서의 처리시간
            return candidate[1][machineId]

        def d_rate(candidate):  # 이 machine에서의 열화율 (operation id "jXX..."의 job 번호로 조회)
            return deterioration_rate[int(candidate[3][1:-2]) - 1]

        if rule in ("SPTSDR", "SPTLDR", "LPTSDR", "LPTLDR"):
            is_sdr, is_spt = rule.endswith("SDR"), rule.startswith("SPT")
            best_candidate, best_key = None, None
            for candidate in filtered_candidates:
                p = candidate[1][machineId]
                d = deterioration_rate[int(candidate[3][1:-2]) - 1]
                cal_p = (p * d) / (1 - d) if is_sdr else p / (d * (1 - d))
                key = (cal_p, p, d) if is_spt else (-cal_p, -p, d)
                if best_key is None or key < best_key:  # 더 좋을 때만 교체 -> 완전 동점이면 앞선 후보 유지
                    best_candidate, best_key = candidate, key
            return best_candidate
        if rule == "SPT":
            return min(filtered_candidates, key=p_time)
        if rule == "LPT":
            return max(filtered_candidates, key=p_time)
        if rule == "SDR":
            return min(filtered_candidates, key=d_rate)
        if rule == "LDR":
            return max(filtered_candidates, key=d_rate)
        return None

    @classmethod
    def dispatching_rule_SDR(cls, candidate_list, machine_list, machineId):
        sorted_candidates = []
        for candidate in candidate_list:
            if candidate[0] == 'Maintenance':
                continue
            job_object, processing_time, _, jop = candidate  # candidate 리스트 구조에 따라 데이터 추출
            job_index = int(jop[1:3]) - 1
            deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
            sorted_candidates.append((candidate, deterioration_rate))
        sorted_candidates.sort(key=lambda x: x[1], reverse=False)
        return [candidate[0] for candidate in sorted_candidates]

    @classmethod
    def dispatching_rule_LDR(cls, candidate_list, machine_list, machineId):
        sorted_candidates = []
        for candidate in candidate_list:
            if candidate[0] == 'Maintenance':
                continue
            job_object, processing_time, _, jop = candidate  # candidate 리스트 구조에 따라 데이터 추출
            job_index = int(jop[1:3]) - 1
            deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
            sorted_candidates.append((candidate, deterioration_rate))
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate[0] for candidate in sorted_candidates]

    @classmethod
    def dispatching_rule_Family_LDR(cls, candidate_list, machine_list, machineId):
        current_machine_family = machine_list[machineId].family_type
        valid_action_list = []
        invalid_action_list = []
        for candidate in candidate_list:
            if candidate[0] == "Maintenance":
                return candidate
            job_object, processing_time, _, jop = candidate
            if job_object.family_type == current_machine_family:
                valid_action_list.append(candidate)
            else:
                invalid_action_list.append(candidate)
        target_candidates = valid_action_list if valid_action_list else invalid_action_list
        sorted_candidates = []
        for candidate in target_candidates:
            job_object, processing_time, _, jop = candidate
            job_index = int(jop[1:3]) - 1  # 예: J01 -> 0
            deterioration_rate = machine_list[machineId].deterioration_rate[job_index]
            sorted_candidates.append((candidate, deterioration_rate))
        sorted_candidates.sort(key=lambda x: x[1], reverse=True)
        return sorted_candidates[0][0]

    @classmethod
    def dispatching_rule_PM(cls, candidate_list):
        candidate = next(filter(lambda item: item[0] == "Maintenance", candidate_list), None)
        return candidate

    @classmethod
    def dispatching_rule_SPT(cls, candidate_list):
        candidate_list.sort(key=lambda x: x[1], reverse=False)
        return candidate_list

    @classmethod
    def dispatching_rule_LPT(cls, candidate_list):
        candidate_list.sort(key=lambda x: x[1], reverse=True)
        return candidate_list

    @classmethod
    def dispatching_rule_SSU(cls, candidate_list):
        candidate_list.sort(key=lambda x: [x[2], x[0].duedate], reverse=False)
        return candidate_list

    @classmethod
    def dispatching_rule_MOR(cls, candidate_list):
        candidate_list.sort(key=lambda x: x[0].remain_operation, reverse=True)
        return candidate_list

    @classmethod
    def dispatching_rule_LOR(cls, candidate_list):
        candidate_list.sort(key=lambda x: x[0].remain_operation, reverse=False)
        return candidate_list

    @classmethod
    def dispatching_rule_EDD(cls, candidate_list):
        candidate_list.sort(key=lambda x: [ x[0].duedate, x[2]] , reverse=False)
        return candidate_list

    @classmethod
    def dispatching_rule_FIFO(cls, candidate_list):
        candidate_list.sort(key=lambda x: [x[0].job_arrival_time, x[2]], reverse=False)
        return candidate_list

    @classmethod
    def dispatching_rule_LIFO(cls, candidate_list):
        candidate_list.sort(key=lambda x: [x[0].job_arrival_time, x[2]], reverse=True)
        return candidate_list

    @classmethod
    def dispatching_rule_SETUP_LPT(cls, candidate_list, machine_list):
        candidate_list.sort(key=lambda x: (getattr(x[0], "family_type", 0), x[1]), reverse=True)
        return candidate_list

    @classmethod
    def dispatching_rule_Family_LPT(cls, candidate_list, machineId):
        same_family = []
        for item in candidate_list:
            if item[0] == "Maintenance":
                same_family.append(item)
                return same_family
        if machineId == "M1":
            for i in candidate_list:
                if i[0].family_type == 1:
                    same_family.append(i)
        elif machineId == "M2":
            for i in candidate_list:
                if i[0].family_type == 2:
                    same_family.append(i)
        elif machineId == "M3":
            for i in candidate_list:
                if i[0].family_type == 3:
                    same_family.append(i)
        elif machineId == "M4":
            for i in candidate_list:
                if i[0].family_type == 4:
                    same_family.append(i)
        for item in candidate_list:
            if item[0] == "Maintenance":
                same_family.append(item)
        same_family.sort(key=lambda x: x[1], reverse=True)
        return same_family