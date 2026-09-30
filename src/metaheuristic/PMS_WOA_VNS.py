import copy
import math
import random
import numpy as np
from src.simlator.Simulator import *
from src.common.pathConfig import *
import matplotlib.pyplot as plt
import time
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class WOA_VNS_PMS:
    population = []
    job_type_list = []
    best_fitness_history = []
    avg_fitness_history = []
    generation = 0

    POP_SIZE = 10
    MAX_ITER = 100
    VNS_MAX_NO_IMPROVE = 4
    THETA = 0.1
    ALPHA = 10.0
    BETA = 0.2
    gantt_on = True

    @classmethod
    def init_population(cls):
        cls.population = []
        for _ in range(cls.POP_SIZE):
            position = cls.init_position_random()
            fitness = cls.evaluate_position(position, gantt=False)
            cls.population.append([position, fitness])

    @classmethod
    def init_position_random(cls):
        n = len(Simulator.get_job_seq())
        m = len(Simulator.machine_list)
        K_MAX = min(m * 4, n)
        dim = n + (K_MAX - m)
        return [random.uniform(1e-8, 1.0) for _ in range(dim)]

    @classmethod
    def decode_position(cls, position):
        """
        논문 원문 인코딩 해석:
        - 앞 n개: job assignment
        - 뒤 k_max-m개: optional maintenance assignment

        Returns:
            machine_jobs: {machine: [job_id, ...]}
            machine_optional_maint: {machine: int}
        """
        job_ids = Simulator.get_job_seq()[:]
        machines = list(Simulator.machine_list)

        n = len(job_ids)
        m = len(machines)

        if len(position) < n:
            raise ValueError(f"position 길이({len(position)})가 job 수({n})보다 작음")

        extra_maint = len(position) - n

        machine_jobs = {machine: [] for machine in machines}
        machine_optional_maint = {machine: 0 for machine in machines}

        # 1) job assignment
        for i in range(n):
            x = position[i]
            machine_idx = int(x * m)
            if machine_idx >= m:
                machine_idx = m - 1
            machine_jobs[machines[machine_idx]].append(job_ids[i])

        # 2) optional maintenance assignment
        for i in range(extra_maint):
            x = position[n + i]

            # (0, 1/(m+1)] -> not assigned
            if x <= 1.0 / (m + 1):
                continue

            bucket = int(x * (m + 1))
            if bucket > m:
                bucket = m

            machine_idx = bucket - 1
            if machine_idx < 0:
                machine_idx = 0
            elif machine_idx >= m:
                machine_idx = m - 1

            machine_optional_maint[machines[machine_idx]] += 1

        return machine_jobs, machine_optional_maint

    @classmethod
    def get_normal_processing_time(cls, job_id, machine_id):
        jobOperId = Simulator.lot_list[job_id].current_operation_id
        return Simulator.Processing_time_table[jobOperId][machine_id]

    @classmethod
    def algorithm_1_grouping(cls, assigned_jobs, k_l, machine):
        """
        논문 Algorithm I의 핵심 구조:
        1) normal processing time 내림차순 정렬
        2) k_l개 그룹으로 균등 배분
        """
        if k_l <= 0:
            raise ValueError("k_l must be >= 1")

        # 1. processing time 내림차순
        sorted_jobs = sorted(
            assigned_jobs,
            key=lambda j: cls.get_normal_processing_time(j, machine),
            reverse=True
        )

        # 2. k_l개 그룹 생성
        groups = [[] for _ in range(k_l)]

        # 논문 Algorithm I 배치 방식
        for x, job in enumerate(sorted_jobs, start=1):
            a = x % k_l
            b = math.ceil(x / k_l)

            if a != 0:
                group_idx = a - 1
            else:
                group_idx = k_l - 1

            groups[group_idx].append(job)

        return groups

    @classmethod
    def cal_setup_time(cls, machine, job):
        base = 70  # 25 30 50 70 70 100 100
        mod = 61  # 17 41 51 51 61 51 101
        A = 7  # machine_family에 곱할 계수
        B = 11  # job_family에 곱할 계수
        machine_family = machine.family_type
        job_family = job.family_type
        if machine_family == job_family:
            return 0
        setup_time = base + ((machine_family * A + job_family * B) % mod)
        return setup_time

    @classmethod
    def compute_machine_makespan(cls, machine, groups, gantt):
        """
        네 원래 열화모델 반영:
        - machine quality가 작업마다 감소
        - maintenance 후 quality = 1.0 복원
        - setup은 machine의 현재 family 상태와 비교
        """
        total_time = 0.0
        current_time = 0.0
        gantt_records = []

        # machine의 초기 setup family 상태
        current_family = Simulator.machine_list[machine]

        # machine의 초기 quality 상태
        current_quality = 1.0

        for group_idx, group in enumerate(groups, start=1):
            group_job_processing_sum = 0.0

            for job_id in group:
                job_obj = Simulator.lot_list[job_id]

                # ---------------------------------
                # 1) sequence-dependent family setup
                # ---------------------------------
                setup_time = cls.cal_setup_time(current_family, job_obj)

                if setup_time > 0:
                    setup_start = current_time
                    setup_end = current_time + setup_time

                    if gantt:
                        gantt_records.append({
                            "machine": machine,
                            "type": "SETUP",
                            "label": f"S:{current_family}->{job_obj.family_type}",
                            "group": group_idx,
                            "start": setup_start,
                            "end": setup_end,
                            "duration": setup_time,
                        })

                    current_time = setup_end

                # ---------------------------------
                # 2) deterioration 반영 processing time
                # ---------------------------------
                base_p = cls.get_normal_processing_time(job_id, machine)

                # quality가 작아질수록 processing time 증가
                real_p = np.round(base_p / current_quality, 3)

                job_start = current_time
                job_end = current_time + real_p

                if gantt:
                    gantt_records.append({
                        "machine": machine,
                        "type": "JOB",
                        "label": job_id,
                        "group": group_idx,
                        "start": job_start,
                        "end": job_end,
                        "duration": real_p,
                        "quality_before": current_quality,
                    })

                current_time = job_end
                group_job_processing_sum += real_p
                # ---------------------------------
                # 3) 작업 후 quality 열화
                # ---------------------------------
                job_index = int(Simulator.lot_list[job_id].job_id[1:]) - 1
                deterioration_rate = np.round(
                    Simulator.machine_list[machine].deterioration_rate[job_index],
                    3
                )
                next_quality = np.round((1 - deterioration_rate) * current_quality, 3)
                current_quality = next_quality

                # 작업 끝나면 machine family 상태 갱신
                current_family = job_obj

            # ---------------------------------
            # 4) maintenance
            # ---------------------------------
            # maintenance_time = cls.ALPHA + cls.BETA * group_job_processing_sum
            if group_idx < len(groups):
                maintenance_time = Hyperparameters.maintenance_time
                pm_start = current_time
                pm_end = current_time + maintenance_time

                if gantt:
                    gantt_records.append({
                        "machine": machine,
                        "type": "PM",
                        "label": "PM",
                        "group": group_idx,
                        "start": pm_start,
                        "end": pm_end,
                        "duration": maintenance_time,
                    })

                current_time = pm_end
                current_quality = 1.0

            total_time = current_time

        if gantt:
            return total_time, gantt_records
        return total_time

    @classmethod
    def evaluate_position(cls, position, gantt):
        machine_jobs, machine_optional_maint = cls.decode_position(position)
        machines = list(Simulator.machine_list)

        machine_makespans = {}
        all_gantt_records = []

        for machine in machines:
            assigned_jobs = machine_jobs[machine]
            k_l = 1 + machine_optional_maint[machine]

            if len(assigned_jobs) == 0:
                machine_makespans[machine] = 0.0
                continue

            groups = cls.algorithm_1_grouping(assigned_jobs, k_l, machine)

            if gantt:
                c_l, gantt_records = cls.compute_machine_makespan(
                    machine, groups, gantt
                )
                all_gantt_records.extend(gantt_records)
            else:
                c_l = cls.compute_machine_makespan(
                    machine, groups, gantt
                )

            machine_makespans[machine] = c_l

        makespan = max(machine_makespans.values()) if machine_makespans else 0.0

        if gantt:
            cls.make_gantt(
                all_gantt_records,
                makespan,
                title="WOA-VNS Paper-style Gantt",
            )
        # print("machine_jobs =", {m: len(v) for m, v in machine_jobs.items()})
        # print("machine_optional_maint =", machine_optional_maint)
        # print("machine_makespans =", machine_makespans)
        return makespan

    @classmethod
    def sort_population(cls):
        cls.population.sort(key=lambda x: x[1], reverse=False)

    @classmethod
    def print_avg_fittness(cls):
        avg_fitness = sum(ind[1] for ind in cls.population) / len(cls.population)
        cls.best_fitness_history.append(cls.population[0][1])
        cls.avg_fitness_history.append(avg_fitness)
        print(f"세대 수 : {cls.generation + 1} 최고 fitness : {cls.population[0][1]} 평균 fitness : {avg_fitness:.3f}")

    @classmethod
    def plot_fitness_curve(cls):
        plt.figure(figsize=(10, 5))
        plt.plot(cls.best_fitness_history, label='Best Makespan', color='blue')
        # plt.plot(cls.avg_fitness_history, label='Average Makespan', color='orange')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.tight_layout()
        if Parameters.save_plot:
            plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Parameters.datasetId}_F{Hyperparameters.family_type}_WOA_VNS_result.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path)
            print(f"WOA-VNS 학습 그래프 저장 완료: {file_path}")
        # plt.show()

    @classmethod
    def make_gantt(cls, gantt_records, makespan, title="WOA-VNS Gantt Chart"):
        if not gantt_records:
            print("간트차트 데이터가 없습니다.")
            return

        plt.figure(figsize=(18, 8))

        machines = sorted(list({rec["machine"] for rec in gantt_records}))
        machine_to_y = {machine: i for i, machine in enumerate(machines)}

        # family type별 색 지정용 팔레트
        family_palette = [
            "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
            "tab:brown", "tab:pink", "tab:olive", "tab:cyan",
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#bcbd22", "#17becf"
        ]
        family_color_map = {}

        def get_job_short_label(job_label):
            raw = str(job_label)

            # 예: M10_J200_F3_..._J15-0 -> J15
            if "_J" in raw:
                tail = raw.split("_J")[-1]
                num = tail.split("-")[0]
                return f"J{num}"

            return raw

        def get_job_family(job_label):
            # gantt_records의 label이 job_id라고 가정
            job_obj = Simulator.lot_list[job_label]
            return job_obj.family_type

        for rec in gantt_records:
            y = machine_to_y[rec["machine"]]
            start = rec["start"]
            duration = rec["duration"]

            if rec["type"] == "JOB":
                family_type = get_job_family(rec["label"])

                if family_type not in family_color_map:
                    family_color_map[family_type] = family_palette[len(family_color_map) % len(family_palette)]

                color = family_color_map[family_type]
                text_color = "white"

            elif rec["type"] == "SETUP":
                color = "gray"
                text_color = "white"

            else:  # PM
                color = "skyblue"
                text_color = "black"

            plt.barh(
                y=y,
                width=duration,
                left=start,
                height=0.5,
                color=color,
                edgecolor="black",
                alpha=0.9
            )

            # # 너무 짧은 막대는 글씨 생략
            # if duration >= 20:
            #     if rec["type"] == "JOB":
            #         label = get_job_short_label(rec["label"])
            #     elif rec["type"] == "SETUP":
            #         label = "SETUP"
            #     else:
            #         label = "PM"
            #
            #     plt.text(
            #         start + duration / 2,
            #         y,
            #         label,
            #         ha="center",
            #         va="center",
            #         fontsize=7,
            #         color=text_color
            #     )

        plt.yticks(range(len(machines)), machines)
        plt.xlabel("Time")
        plt.ylabel("Machine")
        plt.title(f"{title} (Makespan = {makespan:.2f})")
        plt.xlim(0, makespan * 1.02 if makespan > 0 else 1)
        plt.grid(axis="x", linestyle="--", alpha=0.4)

        # family 범례 추가
        import matplotlib.patches as mpatches
        legend_handles = []

        for family_type, color in sorted(family_color_map.items(), key=lambda x: x[0]):
            legend_handles.append(mpatches.Patch(color=color, label=f"Family {family_type}"))

        legend_handles.append(mpatches.Patch(color="gray", label="SETUP"))
        legend_handles.append(mpatches.Patch(color="skyblue", label="PM"))

        plt.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.
        )

        plt.tight_layout()

        gantt_file = (
            f"{Parameters.simulation_time}_"
            f"{Parameters.datasetId}_"
            f"F{Hyperparameters.family_type}_"
            f"WOA_VNS_gantt.png"
        )

        plt.savefig(gantt_file, dpi=300, bbox_inches="tight")
        print(f"간트차트 저장 완료: {gantt_file}")

        # plt.show()
        # plt.close()

    @classmethod
    def setting(cls, data):
        print("simulator on")
        Parameters.set_time_to_string()
        Parameters.set_absolute_path()
        with open(f'{pathConfig.absolute_path}/hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.set_dataSetId([data])
        Hyperparameters.init_hyperparameter_setting(config_data['hyperparameter'])
        Simulator.init_simulator(Parameters.datasetId)
        Simulator.initial_setup_setting()

    @classmethod
    def clip_position(cls, position):
        clipped = []
        for x in position:
            if x <= 0.0:
                clipped.append(1e-8)
            elif x > 1.0:
                clipped.append(1.0)
            else:
                clipped.append(x)
        return clipped

    @classmethod
    def woa_position_update(cls, current_position, best_position, a):
        dim = len(current_position)
        p = random.random()
        r = random.random()

        A = 2 * a * r - a
        C = 2 * r
        if p < 0.5:
            if abs(A) < 1:
                # Encircling prey
                new_position = []
                for d in range(dim):
                    D = abs(C * best_position[d] - current_position[d])
                    x_new = best_position[d] - A * D
                    new_position.append(x_new)
            else:
                # Search for prey
                rand_individual = random.choice(cls.population)
                rand_position = rand_individual[0]

                new_position = []
                for d in range(dim):
                    D = abs(C * rand_position[d] - current_position[d])
                    x_new = rand_position[d] - A * D
                    new_position.append(x_new)
        else:
            # Spiral update
            l = random.uniform(-1, 1)
            b = 1
            new_position = []
            for d in range(dim):
                D_prime = abs(best_position[d] - current_position[d])
                x_new = D_prime * math.exp(b * l) * math.cos(2 * math.pi * l) + best_position[d]
                new_position.append(x_new)
        return cls.clip_position(new_position)

    @classmethod
    def local_search_operator(cls, position):
        new_position = copy.deepcopy(position)
        n = len(Simulator.get_job_seq())
        dim = len(new_position)
        tau = random.randint(0, dim - 1)
        if tau < n:
            # 논문 방식: job 구간이면 [sigma, tau] 뒤집기
            sigma = random.randint(0, tau)
            new_position[sigma:tau + 1] = reversed(new_position[sigma:tau + 1])
        else:
            # 논문 방식: maintenance 구간이면 x_tau = 1 - x_tau
            new_position[tau] = 1.0 - new_position[tau]
        return cls.clip_position(list(new_position))

    @classmethod
    def run_vns_on_best(cls):
        best_position = copy.deepcopy(cls.population[0][0])
        best_fitness = cls.population[0][1]

        E = 1
        while E <= cls.VNS_MAX_NO_IMPROVE:
            candidate_position = cls.local_search_operator(best_position)
            candidate_fitness = cls.evaluate_position(candidate_position, gantt=False)

            if candidate_fitness < best_fitness:
                # print(f"VNS improvement: {best_fitness:.3f} -> {candidate_fitness:.3f}")
                best_position = candidate_position
                best_fitness = candidate_fitness
                E = 1
            else:
                E += 1

        cls.population[0] = [best_position, best_fitness]
        cls.sort_population()

    @classmethod
    def replacement_operator(cls, new_population):
        cls.population = new_population
        cls.sort_population()

    @classmethod
    def search(cls, data):
        cls.population = []
        cls.job_type_list = []
        cls.best_fitness_history = []
        cls.avg_fitness_history = []
        cls.generation = 0

        cls.setting(data)
        cls.init_population()
        print('초기해 생성 완료')
        cls.sort_population()

        while cls.generation < cls.MAX_ITER:
            best_position = copy.deepcopy(cls.population[0][0])
            a = 2 * (1 - cls.generation / cls.MAX_ITER)

            new_population = [copy.deepcopy(cls.population[0])]  # elite

            for i in range(1, cls.POP_SIZE):
                current_position = cls.population[i][0]
                updated_position = cls.woa_position_update(current_position, best_position, a)
                updated_fitness = cls.evaluate_position(updated_position, gantt=False)
                new_population.append([updated_position, updated_fitness])

            cls.population = new_population
            cls.sort_population()
            cls.run_vns_on_best()
            cls.print_avg_fittness()
            cls.generation += 1

        best = cls.population[0]
        cls.plot_fitness_curve()
        print(best)
        if cls.gantt_on:
            cls.evaluate_position(best[0], gantt=True)
        return best[1]

if __name__ == "__main__":
    makespan_list = []
    time_list = []

    start_time = time.time()
    for i in range(5):
        search_start_time = time.time()
        makespan = WOA_VNS_PMS.search("M25_J1000_F7_P10-50_D5-10_Fgap_U1")
        search_end_time = time.time()

        search_elapsed_time = search_end_time - search_start_time
        time_list.append(search_elapsed_time)
        makespan_list.append(makespan)
        print(f"{i + 1}회 반복 종료")

    elapsed_min = (time.time() - start_time) / 60

    makespan_array = np.array(makespan_list)
    mean_makespan = np.mean(makespan_array)
    std_makespan = np.std(makespan_array)

    print(f'Makespan List: {makespan_list}')
    print(f"평균 Makespan: {mean_makespan:.2f}")
    print(f"표준편차: {std_makespan:.2f}")
    print(f"WOA-VNS (소요 시간: {elapsed_min:.2f}분)")

    txt_name = (
        f"{Parameters.simulation_time}_"
        f"{Parameters.datasetId}_"
        f"F{Hyperparameters.family_type}_"
        f"WOA_VNS_result.txt"
    )

    with open(txt_name, "w", encoding="utf-8") as f:
        f.write("===== WOA-VNS Result =====\n")
        f.write(f"makespan list : {makespan_list}\n")
        f.write(f"mean makespan list : {mean_makespan}\n")
        f.write(f"std makespan list : {std_makespan}\n")
        f.write(f"Evaluation Time list (sec): {time_list}\n")

    print(f"📄 Best result TXT 저장 완료: {txt_name}")