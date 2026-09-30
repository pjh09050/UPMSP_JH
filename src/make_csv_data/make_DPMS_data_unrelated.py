import pandas as pd
import numpy as np
import os


def generate_processing_times_and_deterioration_rate(machine_count, job_count, processing_time_range, deterioration_rate_range, operations_per_job, folder_path, problem_num, variation_percent):
    filename_prefix = f"M{machine_count}_J{job_count}_P{processing_time_range[0]}-{processing_time_range[1]}_D{deterioration_rate_range[0]}-{deterioration_rate_range[1]}-var{variation_percent}"

    # 서브 폴더 생성
    target_folder = os.path.join(folder_path, filename_prefix)
    os.makedirs(target_folder, exist_ok=True)

    # 각 job의 기준 처리시간 (평균격)과 편차(±p%) 설정
    variation_percent = variation_percent  # 예: ±10% (0.05=±5%, 0.2=±20%)
    lo, hi = processing_time_range

    # 각 job의 기준 처리시간
    job_processing_times = np.random.randint(lo, hi + 1, job_count)

    # 각 job마다 허용 하한/상한 (범위 밖 벗어나지 않도록 클리핑)
    low_bounds  = np.maximum(lo,  (job_processing_times * (1 - variation_percent)).astype(int))
    high_bounds = np.minimum(hi, np.ceil(job_processing_times * (1 + variation_percent)).astype(int))

    # (job_count, machine_count) 행렬로 기계별 시간 생성 (job별로 다른 범위)
    processing_base = np.random.randint(low_bounds[:, None], high_bounds[:, None] + 1, size=(job_count, machine_count))

    base_processing_times = pd.DataFrame(processing_base, columns=[f'M{i + 1}' for i in range(machine_count)])

    # operations_per_job 만큼 각 job-행을 반복
    processing_times = pd.DataFrame(np.repeat(base_processing_times.values, operations_per_job, axis=0), columns=base_processing_times.columns)
    processing_times.index = [f'J{j + 1:02}{op:02}' for j in range(job_count) for op in range(1, operations_per_job + 1)]

    processing_times_file_path = os.path.join(target_folder, f"{filename_prefix}_U{problem_num}_processing.csv")
    processing_times.to_csv(processing_times_file_path)
    print('Processing times CSV saved')

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
machine_count = 50
job_count = 1000
processing_time_range = (50, 100)
deterioration_rate_range = (5, 10)
operations_per_job = 1
problem_num = 1
variation_percent = 0.05

# problem_num 개의 데이터 인스턴스 생성
for i in range(1, problem_num + 1):
    generate_processing_times_and_deterioration_rate(machine_count, job_count, processing_time_range, deterioration_rate_range, operations_per_job, folder_path, i, variation_percent)