import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_processing_times_and_deterioration_rate(machine_count, job_count, family_type, processing_time_range, deterioration_rate_range, operations_per_job, folder_path, problem_num, machine_variation_percent, job_variation_percent):
    filename_prefix = (
        f"M{machine_count}_J{job_count}_F{family_type}_"
        f"P{processing_time_range[0]}-{processing_time_range[1]}_"
        f"D{deterioration_rate_range[0]}-{deterioration_rate_range[1]}_"
        f"mvar{machine_variation_percent}_jvar{job_variation_percent}"
    )
    # 서브 폴더 생성
    target_folder = os.path.join(folder_path, filename_prefix)
    os.makedirs(target_folder, exist_ok=True)

    lo, hi = processing_time_range

    # 1) family type 할당: 1,2,...,F,1,2,... 순환
    family_num = family_type
    family_types = np.arange(1, family_num + 1)
    families = np.array([family_types[i % family_num] for i in range(job_count)])  # shape: (job_count,)

    # 2) family 기준 처리시간 생성 (각 family마다 하나)
    family_means = np.random.randint(lo, hi + 1, size=family_num)  # shape: (family_num,)

    # 3) 각 job의 기준 처리시간 (family 기준값을 크게 흔들기)
    #    family_means_for_jobs: 각 job이 속한 family의 기준값
    family_means_for_jobs = family_means[families - 1]  # shape: (job_count,)

    # family 기준값에서 ± job_variation_percent 만큼 허용 범위
    job_low_bounds  = np.maximum(lo, (family_means_for_jobs * (1 - job_variation_percent)).astype(int))
    job_high_bounds = np.minimum(hi, np.ceil(family_means_for_jobs * (1 + job_variation_percent)).astype(int))

    # 각 job의 기준 처리시간 (family 내부에서 job 간 편차가 커지는 부분)
    job_processing_times = np.random.randint(job_low_bounds, job_high_bounds + 1)

    # 4) 각 job 안에서 machine 간 처리시간 편차는 작게 (machine_variation_percent 사용)
    mach_low_bounds  = np.maximum(lo, (job_processing_times * (1 - machine_variation_percent)).astype(int))
    mach_high_bounds = np.minimum(hi, np.ceil(job_processing_times * (1 + machine_variation_percent)).astype(int))

    # (job_count, machine_count) 행렬 생성
    processing_base = np.random.randint(mach_low_bounds[:, None], mach_high_bounds[:, None] + 1, size=(job_count, machine_count))
    base_processing_times = pd.DataFrame(processing_base, columns=[f'M{i + 1}' for i in range(machine_count)])

    # operations_per_job 만큼 각 job-행을 반복
    processing_times = pd.DataFrame(np.repeat(base_processing_times.values, operations_per_job, axis=0),columns=base_processing_times.columns)
    processing_index = [f'J{j + 1:02}{op:02}' for j in range(job_count) for op in range(1, operations_per_job + 1)]
    processing_times.index = processing_index

    processing_times_file_path = os.path.join(target_folder, f"{filename_prefix}_U{problem_num}_processing.csv")
    processing_times.to_csv(processing_times_file_path)
    print('Processing times CSV saved')

    # 각 family 내에서의 job 단위 평균/표준편차 (machine 기준)
    job_means = base_processing_times.mean(axis=1)          # shape: (job_count,)
    job_stds  = base_processing_times.std(axis=1, ddof=0)   # 표본이 아니라 모분산 기준이면 ddof=0
    for fam in range(1, family_num + 1):
        mask = (families == fam)
        if not np.any(mask):
            continue  # 해당 family에 job이 없으면 skip

        job_indices = np.where(mask)[0]          # 0-based job index
        job_codes  = [f"J{j+1:02}" for j in job_indices]

        fam_means = job_means[mask]
        fam_stds  = job_stds[mask]

        # 6-1) CSV로 통계 저장
        stats_df = pd.DataFrame({"JobIndex": job_indices + 1,"JobCode": job_codes,"MeanProcessingTime": fam_means,"StdProcessingTime": fam_stds,})

        stats_csv_path = os.path.join(target_folder,f"{filename_prefix}_U{problem_num}_family{fam}_stats.csv")
        stats_df.to_csv(stats_csv_path, index=False)
        print(f'Family {fam} stats CSV saved')

        # 6-2) 그래프 저장 (에러바: 평균 ± 표준편차)
        plt.figure(figsize=(10, 4))
        x = np.arange(len(fam_means))

        plt.errorbar(x,fam_means,yerr=fam_stds,fmt='o',markersize=3, capsize=3)
        plt.xticks(x, job_codes, rotation=90)
        plt.title(f'Family {fam} - Job Processing Time Mean ± Std')
        plt.xlabel('Job')
        plt.ylabel('Processing Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(target_folder,f"{filename_prefix}_U{problem_num}_family{fam}_stats.png")
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f'Family {fam} stats FIG saved')

    # 열화율 생성
    original_deterioration_rates = np.random.uniform(deterioration_rate_range[0], deterioration_rate_range[1], (machine_count, job_count))
    machine_deterioration_rates = 0.01 + ((original_deterioration_rates - 1) / 4) * (0.05 - 0.01)
    machine_deterioration_rates = np.round(machine_deterioration_rates, decimals=3)

    base_deterioration_rates = pd.DataFrame(machine_deterioration_rates.T, columns=[f'M{i + 1}' for i in range(machine_count)])
    deterioration_rates = pd.DataFrame(np.repeat(base_deterioration_rates.values, operations_per_job, axis=0), columns=base_deterioration_rates.columns)
    deterioration_rates.index = [f'J{j + 1:02}{op:02}' for j in range(job_count) for op in range(1, operations_per_job + 1)]

    deterioration_rates_file_path = os.path.join(target_folder, f"{filename_prefix}_U{problem_num}_deterioration.csv")
    deterioration_rates.to_csv(deterioration_rates_file_path)
    print('Deterioration rates CSV saved')
    return f"Files saved in folder: {target_folder}"

folder_path = "../PMS_data"
os.makedirs(folder_path, exist_ok=True)
machine_count = 10
job_count = 300
family_type = 7
processing_time_range = (10, 50)
deterioration_rate_range = (5, 10)
operations_per_job = 1
problem_num = 1
machine_variation_percent = 0.05  # job 내 machine 간 편차 (작게)
job_variation_percent = 0.5       # family 내 job 간 편차 (크게)

# problem_num 개의 데이터 인스턴스 생성
for i in range(1, problem_num + 1):
    generate_processing_times_and_deterioration_rate(machine_count, job_count, family_type, processing_time_range, deterioration_rate_range, operations_per_job, folder_path, i, machine_variation_percent, job_variation_percent)