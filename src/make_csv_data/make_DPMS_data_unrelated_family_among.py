import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_processing_times_and_deterioration_rate(
    machine_count,
    job_count,
    family_type,
    processing_time_range,
    deterioration_rate_range,
    operations_per_job,
    folder_path,
    problem_num
):
    lo, hi = processing_time_range
    family_num = family_type

    # ─────────────────────────────
    # 1) family 할당: 1,2,...,F,1,2,... 순환
    # ─────────────────────────────
    family_ids = np.arange(1, family_num + 1)
    families = np.array([family_ids[i % family_num] for i in range(job_count)])  # shape: (job_count,)

    # ─────────────────────────────
    # 2) family별 center 생성: 랜덤 + "최소 간격" 보장
    # ─────────────────────────────
    total_range = hi - lo
    # 어느 정도 차이 보장용 최소 간격 (너무 작으면 키워도 됨)
    min_required_gap = max(3, total_range // (family_num + 1))

    centers = []
    tries = 0
    max_tries = 10000

    while len(centers) < family_num and tries < max_tries:
        # [lo, hi]에서 랜덤으로 center 뽑기
        c = np.random.randint(lo, hi + 1)
        # 기존 center들과 min_required_gap 이상 차이 나면 채택
        if all(abs(c - x) >= min_required_gap for x in centers):
            centers.append(c)
        tries += 1

    # 혹시 실패하면 fallback: 균등 간격으로 배치
    if len(centers) < family_num:
        step = max(1, (hi - lo) // max(1, (family_num - 1)))
        centers = [lo + i * step for i in range(family_num)]
        centers = [min(hi, c) for c in centers]

    family_centers = np.array(centers, dtype=float)  # shape: (family_num,)

    # ─────────────────────────────
    # 3) family 안에서 job들이 퍼질 반경 (여기서 "family 안 편차" 크게 설정)
    # ─────────────────────────────
    # 전체 범위의 30% 정도를 반경으로 사용 (원하면 0.3 → 0.5 등으로 조절)
    half_width = max(1, int(0.3 * total_range))

    # ─────────────────────────────
    # 4) 파일 이름 prefix (family 간 variation 정보 포함)
    # ─────────────────────────────
    filename_prefix = (
        f"M{machine_count}_J{job_count}_F{family_type}_"
        f"P{processing_time_range[0]}-{processing_time_range[1]}_"
        f"D{deterioration_rate_range[0]}-{deterioration_rate_range[1]}_"
        f"Fgap"
    )

    # 서브 폴더 생성
    target_folder = os.path.join(folder_path, filename_prefix)
    os.makedirs(target_folder, exist_ok=True)

    # ─────────────────────────────
    # 5) job별 family center 기반 구간 설정
    # ─────────────────────────────
    family_centers_for_jobs = family_centers[families - 1]  # shape: (job_count,)

    job_low_bounds = np.maximum(lo, np.floor(family_centers_for_jobs - half_width).astype(int))
    job_high_bounds = np.minimum(hi, np.ceil(family_centers_for_jobs + half_width).astype(int))

    # (job, machine) 처리시간: family별 [low, high] 구간 안에서 랜덤
    processing_base = np.random.randint(
        job_low_bounds[:, None],
        job_high_bounds[:, None] + 1,
        size=(job_count, machine_count)
    )

    base_processing_times = pd.DataFrame(
        processing_base,
        columns=[f'M{i + 1}' for i in range(machine_count)]
    )

    # operations_per_job 만큼 각 job-행을 반복
    processing_times = pd.DataFrame(
        np.repeat(base_processing_times.values, operations_per_job, axis=0),
        columns=base_processing_times.columns
    )
    processing_index = [
        f'J{j + 1:02}{op:02}'
        for j in range(job_count)
        for op in range(1, operations_per_job + 1)
    ]
    processing_times.index = processing_index

    # 처리시간 CSV 저장
    processing_times_file_path = os.path.join(
        target_folder,
        f"{filename_prefix}_U{problem_num}_processing.csv"
    )
    processing_times.to_csv(processing_times_file_path)
    print('Processing times CSV saved')

    # ─────────────────────────────
    # 6) family 정보 CSV 저장
    # ─────────────────────────────
    family_df = pd.DataFrame(
        {
            "Family": np.repeat(families, operations_per_job),
            "FamilyCenter": np.repeat(family_centers_for_jobs, operations_per_job),
            "FamilyLow": np.repeat(job_low_bounds, operations_per_job),
            "FamilyHigh": np.repeat(job_high_bounds, operations_per_job),
        },
        index=processing_index
    )
    family_file_path = os.path.join(
        target_folder,
        f"{filename_prefix}_U{problem_num}_family.csv"
    )
    family_df.to_csv(family_file_path)
    print('Family info CSV saved')

    # ─────────────────────────────
    # 7) 각 family 내 job 평균/표준편차 + 그래프
    # ─────────────────────────────
    job_means = base_processing_times.mean(axis=1)          # shape: (job_count,)
    job_stds  = base_processing_times.std(axis=1, ddof=0)   # 모분산 기준

    for fam in range(1, family_num + 1):
        mask = (families == fam)
        if not np.any(mask):
            continue  # 해당 family에 job이 없으면 skip

        job_indices = np.where(mask)[0]          # 0-based job index
        job_codes  = [f"J{j+1:02}" for j in job_indices]

        fam_means = job_means[mask]
        fam_stds  = job_stds[mask]
        fam_center = family_centers[fam - 1]

        # 7-1) CSV로 통계 저장
        stats_df = pd.DataFrame({
            "JobIndex": job_indices + 1,
            "JobCode": job_codes,
            "MeanProcessingTime": fam_means,
            "StdProcessingTime": fam_stds,
            "FamilyCenter": fam_center
        })

        stats_csv_path = os.path.join(
            target_folder,
            f"{filename_prefix}_U{problem_num}_family{fam}_stats.csv"
        )
        stats_df.to_csv(stats_csv_path, index=False)
        print(f'Family {fam} stats CSV saved')

        # 7-2) 그래프 저장
        plt.figure(figsize=(10, 4))
        x = np.arange(len(fam_means))

        # 기준값 bar (각 job 위치에서 같은 높이의 bar)
        plt.bar(x, [fam_center] * len(fam_means), alpha=0.3, label='Family center')

        # job별 평균 ± 표준편차 (선 없이 작은 점)
        plt.errorbar(
            x,
            fam_means,
            yerr=fam_stds,
            fmt='o',
            markersize=3,
            capsize=3,
            label='Job mean ± std'
        )

        plt.xticks(x, job_codes, rotation=90)
        plt.title(f'Family {fam} - Processing Time (Center & Job Mean ± Std)')
        plt.xlabel('Job')
        plt.ylabel('Processing Time')
        plt.ylim(lo, hi)  # y축 고정

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(
            target_folder,
            f"{filename_prefix}_U{problem_num}_family{fam}_stats.png"
        )
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f'Family {fam} stats FIG saved')

    # ─────────────────────────────
    # 8) 열화율 생성
    # ─────────────────────────────
    original_deterioration_rates = np.random.uniform(
        deterioration_rate_range[0],
        deterioration_rate_range[1],
        (machine_count, job_count)
    )
    machine_deterioration_rates = 0.01 + ((original_deterioration_rates - 1) / 4) * (0.05 - 0.01)
    machine_deterioration_rates = np.round(machine_deterioration_rates, decimals=3)

    base_deterioration_rates = pd.DataFrame(
        machine_deterioration_rates.T,
        columns=[f'M{i + 1}' for i in range(machine_count)]
    )
    deterioration_rates = pd.DataFrame(
        np.repeat(base_deterioration_rates.values, operations_per_job, axis=0),
        columns=base_deterioration_rates.columns
    )
    deterioration_rates.index = processing_index

    deterioration_rates_file_path = os.path.join(
        target_folder,
        f"{filename_prefix}_U{problem_num}_deterioration.csv"
    )
    deterioration_rates.to_csv(deterioration_rates_file_path)
    print('Deterioration rates CSV saved')

    return f"Files saved in folder: {target_folder}"

folder_path = "../PMS_data"
os.makedirs(folder_path, exist_ok=True)

machine_count = 25
job_count = 500
family_type = 5
processing_time_range = (10, 50)
deterioration_rate_range = (8, 13)
operations_per_job = 1
problem_num = 1

for i in range(3, 4):
    print(
        generate_processing_times_and_deterioration_rate(
            machine_count,
            job_count,
            family_type,
            processing_time_range,
            deterioration_rate_range,
            operations_per_job,
            folder_path,
            i
        )
    )
