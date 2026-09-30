import copy
from src.simlator.Simulator import *
import matplotlib.pyplot as plt
import time
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class HGA_PMS:
    population = []
    job_type_list = []
    best_fitness_history = []
    avg_fitness_history = []
    generation = 0  # 현재 세대 수
    POP_SIZE = 15
    # 논문 튜닝값
    D_NN = 0.9  # 최근접 이웃 비율
    D_INS = 0.9  # 삽입 탐색 깊이 (이미 full search라 1.0 유지)
    D_SWAP = 0.9  # 스왑 탐색 비율
    MUTATION_REPEAT = 1

    @classmethod
    def get_total_completion_time(cls, machine_times):
        return sum(machine_times.values())

    @classmethod
    def insertion_sub_neighborhood(cls, arrays, source_m, base_val, base_machine_times):
        machines = list(Simulator.machine_list)

        best_candidate = None
        best_val_cand = float("inf")
        best_machine_times_cand = None

        source_seq = arrays[source_m][:]
        base_total = cls.get_total_completion_time(base_machine_times)

        for act in source_seq:
            if act not in arrays[source_m]:
                continue

            idx = arrays[source_m].index(act)
            reduced_seq = arrays[source_m][:idx] + arrays[source_m][idx + 1:]

            # target machine은 전부 탐색
            for target_m in machines:
                temp_base = copy.deepcopy(arrays)
                temp_base[source_m] = reduced_seq

                positions = cls.get_valid_positions(temp_base[target_m], act)
                if len(positions) == 0:
                    continue

                # position만 일부 샘플링
                k_pos = max(1, int(len(positions) * cls.D_INS))
                sampled_positions = random.sample(positions, min(k_pos, len(positions)))

                for pos in sampled_positions:
                    temp_arr = copy.deepcopy(temp_base)
                    temp_arr[target_m].insert(pos, act)

                    val, machine_times = cls.evaluate_activity_chromosome(temp_arr)

                    # makespan 증가하면 discard
                    if val > base_val:
                        continue

                    cand_total = cls.get_total_completion_time(machine_times)

                    if best_candidate is None:
                        best_candidate = (temp_arr, val, machine_times)
                        best_val_cand = val
                        best_machine_times_cand = machine_times
                    else:
                        best_total = cls.get_total_completion_time(best_machine_times_cand)
                        if (val < best_val_cand) or (val == best_val_cand and cand_total < best_total):
                            best_candidate = (temp_arr, val, machine_times)
                            best_val_cand = val
                            best_machine_times_cand = machine_times

        if best_candidate is None:
            return arrays, base_val, base_machine_times, False

        new_arr, new_val, new_machine_times = best_candidate
        new_total = cls.get_total_completion_time(new_machine_times)
        improved = (new_val < base_val) or (new_val == base_val and new_total < base_total)

        return new_arr, new_val, new_machine_times, improved

    @classmethod
    def swap_sub_neighborhood(cls, arrays, machine, base_val, base_machine_times):
        seq = arrays[machine]
        L = len(seq)

        if L <= 1:
            return arrays, base_val, base_machine_times, False

        base_total = cls.get_total_completion_time(base_machine_times)

        # -------------------------
        # 1) swap pair 일부만 탐색
        # -------------------------
        all_pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
        if len(all_pairs) == 0:
            return arrays, base_val, base_machine_times, False

        k_pairs = max(1, int(len(all_pairs) * cls.D_SWAP))
        sampled_pairs = random.sample(all_pairs, min(k_pairs, len(all_pairs)))

        best_candidate = None
        best_val_cand = float("inf")
        best_machine_times_cand = None

        for i, j in sampled_pairs:
            temp_arr = copy.deepcopy(arrays)
            temp_arr[machine][i], temp_arr[machine][j] = temp_arr[machine][j], temp_arr[machine][i]

            val, machine_times = cls.evaluate_activity_chromosome(temp_arr)

            # makespan 증가하면 discard
            if val > base_val:
                continue

            cand_total = cls.get_total_completion_time(machine_times)

            # -------------------------
            # 2) first improvement
            # -------------------------
            if (val < base_val) or (val == base_val and cand_total < base_total):
                return temp_arr, val, machine_times, True

            # 개선은 아니더라도 후보 저장
            if best_candidate is None:
                best_candidate = (temp_arr, val, machine_times)
                best_val_cand = val
                best_machine_times_cand = machine_times
            else:
                best_total = cls.get_total_completion_time(best_machine_times_cand)
                if (val < best_val_cand) or (val == best_val_cand and cand_total < best_total):
                    best_candidate = (temp_arr, val, machine_times)
                    best_val_cand = val
                    best_machine_times_cand = machine_times

        if best_candidate is None:
            return arrays, base_val, base_machine_times, False

        new_arr, new_val, new_machine_times = best_candidate
        new_total = cls.get_total_completion_time(new_machine_times)
        improved = (new_val < base_val) or (new_val == base_val and new_total < base_total)

        return new_arr, new_val, new_machine_times, improved

    @classmethod
    def nearest_neighbor_sub_neighborhood(cls, arrays, machine, base_val, base_machine_times):
        seq = arrays[machine]
        L = len(seq)

        if L <= 1:
            return arrays, base_val, base_machine_times, False

        base_total = cls.get_total_completion_time(base_machine_times)

        # -------------------------
        # 1) 일부 index만 탐색
        # -------------------------
        indices = list(range(L))
        k_idx = max(1, int(len(indices) * cls.D_NN))
        sampled_indices = random.sample(indices, min(k_idx, len(indices)))

        best_candidate = None
        best_val_cand = float("inf")
        best_machine_times_cand = None

        for i in sampled_indices:
            if i >= len(arrays[machine]):
                continue

            act = arrays[machine][i]
            reduced_seq = arrays[machine][:i] + arrays[machine][i + 1:]

            # -------------------------
            # 2) 인접 위치만 탐색
            # -------------------------
            candidate_positions = []
            for pos in [i - 1, i, i + 1]:
                if 0 <= pos <= len(reduced_seq):
                    candidate_positions.append(pos)

            candidate_positions = list(set(candidate_positions))

            # PM 제약 반영
            valid_positions = cls.get_valid_positions(reduced_seq, act)
            candidate_positions = [pos for pos in candidate_positions if pos in valid_positions]

            if len(candidate_positions) == 0:
                continue

            for pos in candidate_positions:
                temp_arr = copy.deepcopy(arrays)
                temp_arr[machine] = reduced_seq[:]
                temp_arr[machine].insert(pos, act)

                val, machine_times = cls.evaluate_activity_chromosome(temp_arr)

                # makespan 증가하면 discard
                if val > base_val:
                    continue

                cand_total = cls.get_total_completion_time(machine_times)

                # -------------------------
                # 3) first improvement
                # -------------------------
                if (val < base_val) or (val == base_val and cand_total < base_total):
                    return temp_arr, val, machine_times, True

                # 개선은 아니더라도 best candidate 저장
                if best_candidate is None:
                    best_candidate = (temp_arr, val, machine_times)
                    best_val_cand = val
                    best_machine_times_cand = machine_times
                else:
                    best_total = cls.get_total_completion_time(best_machine_times_cand)
                    if (val < best_val_cand) or (val == best_val_cand and cand_total < best_total):
                        best_candidate = (temp_arr, val, machine_times)
                        best_val_cand = val
                        best_machine_times_cand = machine_times

        if best_candidate is None:
            return arrays, base_val, base_machine_times, False

        new_arr, new_val, new_machine_times = best_candidate
        new_total = cls.get_total_completion_time(new_machine_times)
        improved = (new_val < base_val) or (new_val == base_val and new_total < base_total)

        return new_arr, new_val, new_machine_times, improved

    @classmethod
    def local_search(cls, arrays):
        current_arr = copy.deepcopy(arrays)
        current_val, current_machine_times = cls.evaluate_activity_chromosome(current_arr)

        # =========================
        # 1. Insertion neighborhood
        # =========================
        machines = list(Simulator.machine_list)
        sorted_machines = sorted(
            machines,
            key=lambda m: current_machine_times[m],
            reverse=True
        )
        k_machines = max(1, int(len(sorted_machines) * cls.D_INS))
        target_machines = sorted_machines[:k_machines]

        for m in target_machines:
            new_arr, new_val, new_machine_times, improved = cls.insertion_sub_neighborhood(current_arr, m, current_val, current_machine_times)
            if improved:
                current_arr = new_arr
                current_val = new_val
                current_machine_times = new_machine_times

        # =========================
        # 2. Swap neighborhood
        # =========================
        sorted_machines = sorted(
            machines,
            key=lambda m: current_machine_times[m],
            reverse=True
        )
        k_machines = max(1, int(len(sorted_machines) * cls.D_SWAP))
        target_machines = sorted_machines[:k_machines]

        for m in target_machines:
            new_arr, new_val, new_machine_times, improved = cls.swap_sub_neighborhood(
                current_arr, m, current_val, current_machine_times
            )
            if improved:
                current_arr = new_arr
                current_val = new_val
                current_machine_times = new_machine_times

        # =========================
        # 3. Nearest neighbor search
        # =========================
        sorted_machines = sorted(
            machines,
            key=lambda m: current_machine_times[m],
            reverse=True
        )
        k_machines = max(1, int(len(sorted_machines) * cls.D_NN))
        target_machines = sorted_machines[:k_machines]

        for m in target_machines:
            new_arr, new_val, new_machine_times, improved = cls.nearest_neighbor_sub_neighborhood(
                current_arr, m, current_val, current_machine_times
            )
            if improved:
                current_arr = new_arr
                current_val = new_val
                current_machine_times = new_machine_times

        return current_arr, current_val

    @classmethod
    def insert_best_position_same_machine(cls, arrays, machine, act):
        positions = cls.get_valid_positions(arrays[machine], act)

        if len(positions) == 0:
            return arrays

        best_pos = None
        best_val = float("inf")

        for pos in positions:
            temp_arr = copy.deepcopy(arrays)
            temp_arr[machine].insert(pos, act)

            val, _ = cls.evaluate_activity_chromosome(temp_arr)

            if val < best_val:
                best_val = val
                best_pos = pos

        arrays[machine].insert(best_pos, act)
        return arrays

    @classmethod
    def worst_machine_mutation(cls, chromosome):
        arrays = copy.deepcopy(chromosome[0])
        machines = list(Simulator.machine_list)
        _, machine_times = cls.evaluate_activity_chromosome(arrays)
        worst_m = max(machine_times, key=machine_times.get)
        removed_activities = arrays[worst_m][:]
        arrays[worst_m] = []
        for act in removed_activities:
            # PM은 자기 machine 유지
            if act[0] == "PM":
                target_m = worst_m
            else:
                candidate_machines = [m for m in machines if m != worst_m]
                if len(candidate_machines) == 0:
                    candidate_machines = [worst_m]
                target_m = random.choice(candidate_machines)
            arrays = cls.insert_best_position_same_machine(arrays, target_m, act)
        fitness, _ = cls.evaluate_activity_chromosome(arrays)
        return [arrays, fitness]

    @classmethod
    def build_lsec_offspring(cls, parent1_arr, parent2_arr, cut_points, mode="prefix"):
        machines = list(Simulator.machine_list)
        child_arr = {m: [] for m in machines}

        # =========================
        # 1. parent1 일부 복사
        # =========================
        for m in machines:
            seq1 = parent1_arr[m]
            cut = cut_points[m]

            if len(seq1) == 0:
                child_arr[m] = []
                continue

            if mode == "prefix":
                child_arr[m] = copy.deepcopy(seq1[:cut])
            else:  # suffix
                child_arr[m] = copy.deepcopy(seq1[cut:])

        # 현재 offspring에 들어간 activity 추적
        existing = set()
        for m in machines:
            for act in child_arr[m]:
                existing.add((act[0], act[1]))

        # =========================
        # 2. parent2의 missing activity를
        #    같은 machine 안에 best insertion
        # =========================
        for m in machines:
            for act in parent2_arr[m]:
                key = (act[0], act[1])
                if key in existing:
                    continue
                if act[0] == "PM":
                    target_m = act[1].replace("PM_", "")
                else:
                    target_m = m
                child_arr = cls.insert_best_position_same_machine(child_arr, target_m, act)
                existing.add(key)

        return child_arr

    @classmethod
    def LSEC_crossover(cls, mom, dad):
        mom_arr = mom[0]
        dad_arr = dad[0]

        machines = list(Simulator.machine_list)

        # machine마다 crossover point 선택
        cut_points = {}
        for m in machines:
            L = len(mom_arr[m])

            if L == 0:
                cut_points[m] = 0
            else:
                # prefix는 [:cut], suffix는 [cut:]로 쓸 거라
                # cut은 0~L 사이 값 가능
                cut_points[m] = random.randint(0, L)

        # offspring1: parent1 prefix + parent2 missing insertion
        child1_arr = cls.build_lsec_offspring(
            parent1_arr=mom_arr,
            parent2_arr=dad_arr,
            cut_points=cut_points,
            mode="prefix"
        )

        # # offspring2: parent1 suffix + parent2 missing insertion
        # child2_arr = cls.build_lsec_offspring(
        #     parent1_arr=mom_arr,
        #     parent2_arr=dad_arr,
        #     cut_points=cut_points,
        #     mode="suffix"
        # )

        fitness1, _ = cls.evaluate_activity_chromosome(child1_arr)
        # fitness2, _ = cls.evaluate_activity_chromosome(child2_arr)

        return [child1_arr, fitness1]

    @staticmethod
    def get_valid_positions(seq, act):
        L = len(seq)
        valid_positions = []

        for pos in range(L + 1):
            if act[0] == "PM":
                # PM 연속 금지
                if pos > 0 and seq[pos - 1][0] == "PM":
                    continue
                if pos < L and seq[pos][0] == "PM":
                    continue
            valid_positions.append(pos)
        return valid_positions

    @classmethod
    def init_activity_chromosome(cls):
        machines = list(Simulator.machine_list)
        jobs = Simulator.get_job_seq()
        # machine당 PM 1개
        pm_ids = [f"PM_{m}" for m in machines]
        # 전체 activity 목록
        cls.ALL_ACTIVITIES = (
                [("JOB", j) for j in jobs] +
                [("PM", p) for p in pm_ids]
        )
        arrays = {m: [] for m in machines}

        shuffled_jobs = jobs[:]
        random.shuffle(shuffled_jobs)
        for j in shuffled_jobs:
            m = random.choice(machines)
            arrays[m].append(("JOB", j))

        for m, pm_id in zip(machines, pm_ids):
            act = ("PM", pm_id)
            positions = cls.get_valid_positions(arrays[m], act)
            pos = random.choice(positions)
            arrays[m].insert(pos, act)
        return arrays

    @classmethod
    def init_population(cls):
        for i in range(cls.POP_SIZE):
            arrays = cls.init_activity_chromosome()
            fitness, _ = cls.evaluate_activity_chromosome(arrays, False)
            cls.population.append([arrays, fitness])

    @classmethod
    def sort_population(cls):
        cls.population.sort(key=lambda x: x[1], reverse=False)

    @classmethod
    def selection_operator(cls):
        mom_ch = random.choice(cls.population)
        dad_ch = random.choice(cls.population)
        while dad_ch == mom_ch and len(cls.population) > 1:
            dad_ch = random.choice(cls.population)
        return mom_ch, dad_ch

    @classmethod
    def replacement_operator(cls, child):
        # 1. 중복 체크
        if any(child[0] == p[0] for p in cls.population):
            return
        # 2. worst와 비교
        worst = cls.population[-1]
        if child[1] < worst[1]:
            cls.population.pop()
            cls.population.append(child)

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
    def evaluate_activity_chromosome(cls, arrays, gantt=False):

        machines = list(Simulator.machine_list)
        machine_makespans = {}
        all_gantt_records = []

        for machine in machines:

            seq = arrays[machine]

            if len(seq) == 0:
                machine_makespans[machine] = 0.0
                continue

            current_time = 0.0
            current_family = Simulator.machine_list[machine]
            current_quality = 1.0
            gantt_records = []

            for activity_type, job_id in seq:

                # =========================
                # JOB 처리
                # =========================
                if activity_type == "JOB":

                    job_obj = Simulator.lot_list[job_id]

                    # setup
                    setup_time = cls.cal_setup_time(current_family, job_obj)

                    if setup_time > 0:
                        setup_start = current_time
                        setup_end = current_time + setup_time

                        if gantt:
                            gantt_records.append({
                                "machine": machine,
                                "type": "SETUP",
                                "label": f"S:{current_family.family_type}->{job_obj.family_type}",
                                "start": setup_start,
                                "end": setup_end,
                                "duration": setup_time,
                            })

                        current_time = setup_end

                    # processing
                    jobOperId = job_obj.current_operation_id
                    base_p = Simulator.Processing_time_table[jobOperId][machine]
                    real_p = np.round(base_p / current_quality, 3)

                    job_start = current_time
                    job_end = current_time + real_p

                    if gantt:
                        gantt_records.append({
                            "machine": machine,
                            "type": "JOB",
                            "label": job_id,
                            "start": job_start,
                            "end": job_end,
                            "duration": real_p,
                        })

                    current_time = job_end

                    # deterioration
                    job_index = int(job_obj.job_id[1:]) - 1
                    deterioration_rate = np.round(
                        Simulator.machine_list[machine].deterioration_rate[job_index], 3
                    )
                    current_quality = np.round((1 - deterioration_rate) * current_quality, 3)

                    current_family = job_obj

                # =========================
                # PM 처리
                # =========================
                elif activity_type == "PM":

                    pm_start = current_time
                    pm_end = current_time + Hyperparameters.maintenance_time

                    if gantt:
                        gantt_records.append({
                            "machine": machine,
                            "type": "PM",
                            "label": "PM",
                            "start": pm_start,
                            "end": pm_end,
                            "duration": Hyperparameters.maintenance_time,
                        })

                    current_time = pm_end
                    current_quality = 1.0

            machine_makespans[machine] = current_time

            if gantt:
                all_gantt_records.extend(gantt_records)

        makespan = max(machine_makespans.values()) if machine_makespans else 0.0

        if gantt:
            cls.make_gantt(
                all_gantt_records,
                makespan,
                title="HGA Activity-based Gantt"
            )

        # 🔥 핵심 변경 (여기)
        return makespan, machine_makespans

    @classmethod
    def print_avg_fittness(cls):
        sum_fitness = sum(i[1] for i in cls.population)
        avg_fitness = sum_fitness / len(cls.population)
        cls.best_fitness_history.append(cls.population[0][1])
        cls.avg_fitness_history.append(avg_fitness)
        print(f"세대 수 : {cls.generation+1} 최고 fitness : {cls.population[0][1]} 평균 fitness : {avg_fitness:.3f}")

    @classmethod
    def make_gantt(cls, gantt_records, makespan, title="GA Gantt Chart"):
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
            f"GA_gantt.png"
        )

        plt.savefig(gantt_file, dpi=300, bbox_inches="tight")
        print(f"간트차트 저장 완료: {gantt_file}")

        # plt.show()
        # plt.close()

    @classmethod
    def plot_fitness_curve(cls):
        plt.figure(figsize=(10, 5))
        plt.plot(cls.best_fitness_history, label='Best Makespan', color='blue')
        plt.plot(cls.avg_fitness_history, label='Average Makespan', color='orange')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.tight_layout()
        if Parameters.save_plot:
            plot_save_directory = f"{pathConfig.simulator_result_plot_save_path}{os.sep}"
            result_name = f"{Parameters.simulation_time}_{Hyperparameters.object}_{Parameters.datasetId}_F{Hyperparameters.family_type}_GA_result.png"
            file_path = os.path.join(plot_save_directory, result_name)
            plt.savefig(file_path)
            print(f"GA 학습 그래프 저장 완료: {file_path}")
        # plt.show()

    @classmethod
    def setting(cls, data):
        print("simulator on")
        Parameters.set_time_to_string()  # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_absolute_path()
        with open(f'{pathConfig.absolute_path}/hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.set_dataSetId([data])
        Hyperparameters.init_hyperparameter_setting(config_data['hyperparameter'])  # 시뮬레이터의 하이퍼파라미터 설정
        Simulator.init_simulator(Parameters.datasetId)  # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정
        Simulator.initial_setup_setting()

    @classmethod
    def search(cls, data):
        cls.population = []
        cls.job_type_list = []
        cls.best_fitness_history = []
        cls.avg_fitness_history = []
        cls.generation = 0

        cls.setting(data)

        # =========================
        # 시간 기반 종료조건 설정
        # =========================
        num_jobs = len(Simulator.get_job_seq())
        num_machines = len(list(Simulator.machine_list))
        num_pm = num_machines  # 지금 네 구현은 machine당 PM 1개

        time_limit = 0.125 * num_jobs * num_machines * num_pm  # seconds
        start_time = time.time()

        cls.init_population()
        print('초기해 생성 완료')
        cls.sort_population()
        while True:
            mom_ch, dad_ch = cls.selection_operator()
            offspring1 = cls.LSEC_crossover(mom_ch, dad_ch)
            offspring1 = cls.worst_machine_mutation(offspring1)
            arrays1, fitness1 = cls.local_search(offspring1[0])
            offspring1 = [arrays1, fitness1]
            cls.replacement_operator(offspring1)
            cls.sort_population()
            cls.print_avg_fittness()
            cls.generation += 1

            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                best = cls.population[0]
                fitness, _ = cls.evaluate_activity_chromosome(best[0], True)
                print(f"시간 종료: {elapsed:.2f}s / limit={time_limit:.2f}s")
                break

        cls.plot_fitness_curve()
        print(cls.population[0])
        return cls.population[0][-1]


dataset_list = ["M10_J500_F7_P10-50_D5-10_Fgap_U1"]
for dataset in dataset_list:
    start_time = time.time()
    makespan_list = []
    time_list = []
    for i in range(5):
        search_start_time = time.time()
        makespan = HGA_PMS.search(dataset)
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
    print(f"GA (소요 시간: {elapsed_min:.2f}분)")

    txt_name = (
        f"{Parameters.simulation_time}_"
        f"{Parameters.datasetId}_"
        f"F{Hyperparameters.family_type}_"
        f"Hybrid GA_result_{time.time()}.txt"
    )

    with open(txt_name, "w", encoding="utf-8") as f:
        f.write("===== Hybrid GA Result =====\n")
        f.write(f"makespan list : {makespan_list}\n")
        f.write(f"mean makespan list : {mean_makespan:.2f}\n")
        f.write(f"std makespan list : {std_makespan:.2f}\n")
        f.write(f"Evaluation Time list (sec): {time_list}\n")
    print(f"📄 Best result TXT 저장 완료: {txt_name}")