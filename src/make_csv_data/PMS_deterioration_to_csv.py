import csv
import os

def get_csv(_file_name):
    # Define the file path
    file_path = "C:/Users/user/Desktop/SynologyDrive/연구실/스케쥴링/데이터셋/pms_job 열화/OB_Problem_OptSolution/"
    file_name = _file_name
    text_file = os.path.join(file_path, file_name)

    # Read the text file
    with open(text_file, 'r') as f:
        text_data = f.read()

    # Split the data into lines
    lines = list(map(float, text_data.strip().split()))

    # Extract job and machine counts
    jobs = int(lines[0])
    machines = int(lines[1])

    # Initialize starting index
    index = 2

    # Extract processing times for each machine
    processing_times = []
    for _ in range(machines):
        processing_times.append(lines[index:index + jobs])
        index += jobs

    # Extract deterioration rates for each machine
    deterioration_rates = []
    for _ in range(machines):
        deterioration_rates.append(lines[index:index + jobs])
        index += jobs

    # Create dynamic headers
    headers = [''] + [f'M{i+1}' for i in range(machines)]

    # Construct the CSV file paths
    save_path = "C:/Users/user/Desktop/SynologyDrive/연구실/스케쥴링/데이터셋/pms_job 열화/"
    csv_file_path_processing = os.path.join(save_path, f'{file_name}_processing.csv')
    csv_file_path_deterioration = os.path.join(save_path, f'{file_name}_deterioration.csv')

    # Create CSV for processing times
    with open(csv_file_path_processing, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(headers)

        # Write data
        for i in range(jobs):
            job_id = f'J{i+1:02d}'
            row = [job_id] + [processing_times[m][i] for m in range(machines)]
            writer.writerow(row)

    # Create CSV for deterioration rates
    with open(csv_file_path_deterioration, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(headers)

        # Write data
        for i in range(jobs):
            job_id = f'{i+1:02d}'
            row = [job_id] + [deterioration_rates[m][i] for m in range(machines)]
            writer.writerow(row)

    print(f"CSV files have been created at: {csv_file_path_processing} and {csv_file_path_deterioration}")

file_name = 'Ni_8_2-1_1_1'
get_csv(file_name)
