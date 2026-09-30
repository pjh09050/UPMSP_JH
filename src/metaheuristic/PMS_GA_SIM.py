import copy
from src.simlator.Simulator import *
import matplotlib.pyplot as plt
import time
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class GA_PMS:
    population = []
    job_type_list = []
    best_fitness_history = []
    avg_fitness_history = []
    generation = 0  # 현재 세대 수
    MUTATION_PROB = 0.15
    CROSSOVER_PROB = 0.75
    POP_SIZE = 10
    NUM_OFFSPRING = 10

    @classmethod
    def init_job_seq_random(cls):
        job_seq = Simulator.get_job_seq()
        random.shuffle(job_seq)
        return job_seq

    @classmethod
    def init_mac_seq_random(cls, job_seq):
        all_machines = list(Simulator.machine_list)
        machine_seq = [random.choice(all_machines) for _ in job_seq]
        return machine_seq

    # @classmethod
    # def init_pm_seq_random(cls, job_seq):
    #     pm_seq = [random.randint(0, 1) for _ in job_seq]
    #     return pm_seq

    @classmethod
    def init_pm_seq_random(cls, job_seq, mac_seq):
        pm_seq = [0] * len(job_seq)
        for i in range(len(job_seq)):
            if i > 0 and mac_seq[i] == mac_seq[i - 1] and pm_seq[i - 1] == 1:
                pm_seq[i] = 0  # 이전 job이 같은 machine에서 PM이면 이번 job은 PM 금지
            else:
                pm_seq[i] = random.randint(0, 1)
        return pm_seq

    @classmethod
    def init_population(cls):
        for i in range(cls.POP_SIZE):
            job_seq = cls.init_job_seq_random()
            mac_seq = cls.init_mac_seq_random(job_seq)
            pm_seq = cls.init_pm_seq_random(job_seq, mac_seq)
            cls.job_type_list = list(set(job_seq))
            fittness = Simulator.get_fittness_with_meta_heuristic(job_seq, mac_seq, pm_seq, False)
            # print(f"{i}번째 해, fitness: {fittness}, job_seq: {job_seq}, mac_seq: {mac_seq}, pm_seq: {pm_seq}")
            cls.population.append([job_seq, mac_seq, pm_seq, fittness])

    @classmethod
    def validate_pm_sequence(cls, job_seq, mac_seq, pm_seq):
        machine_indices = {}  # 각 machine마다 job index를 모음
        for idx, mac in enumerate(mac_seq):
            if mac not in machine_indices:
                machine_indices[mac] = []
            machine_indices[mac].append(idx)
        # machine별로 순회하면서 연속 PM 여부 검사
        for mac, indices in machine_indices.items():
            prev_pm = 0
            for idx in indices:
                if pm_seq[idx] == 1 and prev_pm == 1:
                    pm_seq[idx] = 0  # 같은 machine에서 연속 PM 발생 시 후행 PM 제거
                prev_pm = pm_seq[idx]
        return pm_seq

    @classmethod
    def crossover_operator_3segment(cls, mom, dad):
        mom_job, mom_mac, mom_pm = mom[0], mom[1], mom[2]
        dad_job, dad_mac, dad_pm = dad[0], dad[1], dad[2]
        length = len(mom_job)

        ### Segment 1: Position-Based Crossover (PBC) for job_seq
        point_count = random.randint(1, length // 2)
        crossover_points = sorted(random.sample(range(length), point_count))

        child1_job = [-1] * length
        for idx in crossover_points:
            child1_job[idx] = mom_job[idx]

        def fill_remaining(child, other_parent, fixed_points):
            pos = 0
            for gene in other_parent:
                if gene not in fixed_points:
                    while child[pos] != -1:
                        pos += 1
                    child[pos] = gene
            return child

        child1_job = fill_remaining(child1_job, dad_job, set(child1_job))

        # Segment 2: Uniform crossover for pm_seq
        child1_pm = [random.choice([m, d]) for m, d in zip(mom_pm, dad_pm)]

        # Segment 3: Uniform crossover for mac_seq
        child1_mac = [random.choice([m, d]) for m, d in zip(mom_mac, dad_mac)]

        # Fitness 계산
        validate_child1_pm = cls.validate_pm_sequence(child1_job, child1_mac, child1_pm)
        fitness1 = Simulator.get_fittness_with_meta_heuristic(child1_job, child1_mac, validate_child1_pm, False)
        offspring1 = [child1_job, child1_mac, validate_child1_pm, fitness1]

        # fitness1 = Simulator.get_fittness_with_meta_heuristic(child1_job, child1_mac, child1_pm, False)
        # offspring1 = [child1_job, child1_mac, child1_pm, fitness1]
        return offspring1

    @classmethod
    def mutation_operator_3segment(cls, chromosome):
        job_seq, mac_seq, pm_seq = chromosome[0], chromosome[1], chromosome[2]
        length = len(job_seq)

        # Segment 1: job_seq – Inversion Mutation
        i, j = sorted(random.sample(range(length), 2))
        job_seq[i:j + 1] = list(reversed(job_seq[i:j + 1]))

        # Segment 2: pm_seq – Insertion Mutation
        i = random.randint(0, length - 1)
        pm_seq.pop(i)
        bit = random.randint(0, 1)
        j = random.randint(0, length - 2)
        pm_seq.insert(j, bit)

        # Segment 3: mac_seq – Reciprocal Exchange Mutation (twice)
        i1, j1 = random.sample(range(length), 2)
        mac_seq[i1], mac_seq[j1] = mac_seq[j1], mac_seq[i1]
        i2, j2 = random.sample(range(length), 2)
        mac_seq[i2], mac_seq[j2] = mac_seq[j2], mac_seq[i2]

        validate_pm_seq = cls.validate_pm_sequence(job_seq, mac_seq, pm_seq)
        fitness = Simulator.get_fittness_with_meta_heuristic(job_seq, mac_seq, validate_pm_seq, False)
        mutated = [job_seq, mac_seq, validate_pm_seq, fitness]

        # fitness = Simulator.get_fittness_with_meta_heuristic(job_seq, mac_seq, pm_seq, False)
        # mutated = [job_seq, mac_seq, pm_seq, fitness]
        return mutated

    @classmethod
    def sort_population(cls):
        cls.population.sort(key=lambda x: x[3], reverse=False)

    @classmethod
    def selection_operator(cls, k=10):
        # 첫 번째 부모 선택
        candidates1 = random.sample(cls.population, k)
        candidates1.sort(key=lambda x: x[3])
        mom_ch = candidates1[0]
        # 두 번째 부모 선택
        candidates2 = random.sample(cls.population, k)
        candidates2.sort(key=lambda x: x[3])
        dad_ch = candidates2[0]
        return mom_ch, dad_ch

    @classmethod
    def replacement_operator(cls, offsprings):
        for _ in range(cls.NUM_OFFSPRING):
            cls.population.pop()
        cls.population.extend(offsprings)

    @classmethod
    def print_avg_fittness(cls):
        sum_fitness = sum(i[3] for i in cls.population)
        avg_fitness = sum_fitness / len(cls.population)
        cls.best_fitness_history.append(cls.population[0][3])
        cls.avg_fitness_history.append(avg_fitness)
        print(f"세대 수 : {cls.generation+1} 최고 fitness : {cls.population[0][3]} 평균 fitness : {avg_fitness:.3f}")

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
        cls.init_population()
        print('초기해 생성 완료')
        cls.sort_population()
        while True:
            offsprings = []
            for i in range(cls.NUM_OFFSPRING):
                mom_ch, dad_ch = cls.selection_operator()
                if random.random() < cls.CROSSOVER_PROB:
                    offspring1 = cls.crossover_operator_3segment(mom_ch, dad_ch)
                else:
                    offspring1 = copy.deepcopy(mom_ch)
                if random.random() < cls.MUTATION_PROB:
                    offspring1 = cls.mutation_operator_3segment(offspring1)
                offsprings.append(offspring1)

            cls.replacement_operator(offsprings)
            cls.sort_population()
            cls.print_avg_fittness()
            cls.generation += 1
            if cls.generation == 2:
                best = cls.population[0]
                Simulator.get_fittness_with_meta_heuristic(best[0], best[1], best[2], True)
                break
        cls.plot_fitness_curve()
        print(cls.population[0])
        return cls.population[0][-1]

makespan_list = []
time_list = []
start_time = time.time()
for i in range(2):
    search_start_time = time.time()
    makespan = GA_PMS.search("M10_J200_F3_P10-50_D5-10_Fgap_U1")
    search_end_time = time.time()
    search_elapsed_time = search_end_time - search_start_time
    time_list.append(search_elapsed_time)
    makespan_list.append(makespan)
    print(f"{i+1}회 반복 종료")
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
    f"GA_result.txt"
)

with open(txt_name, "w", encoding="utf-8") as f:
    f.write("===== Second Stage Result =====\n")
    f.write(f"makespan list : {makespan_list}\n")
    f.write(f"mean makespan list : {mean_makespan:}\n")
    f.write(f"std makespan list : {std_makespan}\n")
    f.write(f"Evaluation Time list (sec): {time_list:}\n")
print(f"📄 Best result TXT 저장 완료: {txt_name}")