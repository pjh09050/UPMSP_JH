import pandas as pd
import numpy as np
import os


def generate_processing_times_and_deterioration_rate(machine_count, job_count, processing_time_range, deterioration_rate_range, operations_per_job, folder_path, problem_num):
    filename_prefix = f"M{machine_count}_J{job_count}_P{processing_time_range[0]}-{processing_time_range[1]}_D{deterioration_rate_range[0]}-{deterioration_rate_range[1]}"

    # 서브 폴더 생성
    target_folder = os.path.join(folder_path, filename_prefix)
    os.makedirs(target_folder, exist_ok=True)

    # Generate processing times for each job (same value for all machines)
    job_processing_times = np.random.randint(processing_time_range[0], processing_time_range[1] + 1, job_count)
    base_processing_times = pd.DataFrame({f'M{i + 1}': job_processing_times for i in range(machine_count)})

    processing_times = pd.DataFrame(np.repeat(base_processing_times.values, operations_per_job, axis=0),columns=base_processing_times.columns)
    processing_times.index = [f'J{j + 1:02}{op:02}' for j in range(job_count) for op in range(1, operations_per_job + 1)]

    processing_times_file_path = os.path.join(target_folder, f"{filename_prefix}_ID{problem_num}_processing.csv")
    processing_times.to_csv(processing_times_file_path)
    print('Processing times CSV saved')

    # 열화율 생성
    original_deterioration_rates = np.random.uniform(deterioration_rate_range[0], deterioration_rate_range[1], (machine_count, job_count))
    machine_deterioration_rates = 0.01 + ((original_deterioration_rates - 1) / 4) * (0.05 - 0.01)
    machine_deterioration_rates = np.round(machine_deterioration_rates, decimals=3)

    base_deterioration_rates = pd.DataFrame(machine_deterioration_rates.T, columns=[f'M{i + 1}' for i in range(machine_count)])
    deterioration_rates = pd.DataFrame(np.repeat(base_deterioration_rates.values, operations_per_job, axis=0), columns=base_deterioration_rates.columns)
    deterioration_rates.index = [f'J{j + 1:02}{op:02}' for j in range(job_count) for op in range(1, operations_per_job + 1)]

    deterioration_rates_file_path = os.path.join(target_folder, f"{filename_prefix}_ID{problem_num}_deterioration.csv")
    deterioration_rates.to_csv(deterioration_rates_file_path)
    print('Deterioration rates CSV saved')
    return f"Files saved in folder: {target_folder}"

folder_path = "../PMS_data"
os.makedirs(folder_path, exist_ok=True)
machine_count = 10
job_count = 200
processing_time_range = (10, 100)
deterioration_rate_range = (5, 10)
operations_per_job = 1
problem_num = 1

# problem_num 개의 데이터 인스턴스 생성
for i in range(1, problem_num + 1):
    generate_processing_times_and_deterioration_rate(machine_count, job_count, processing_time_range, deterioration_rate_range, operations_per_job, folder_path, i)