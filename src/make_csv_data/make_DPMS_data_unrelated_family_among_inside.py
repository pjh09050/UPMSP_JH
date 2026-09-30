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
    problem_num,
    machine_variation_percent,
    job_variation_percent,   # 🔥 family 안 job 편차 세기 조절
):
    lo, hi = processing_time_range
    total_range = hi - lo
    family_num = family_type

    # 1) family 할당 (1,2,...,F 반복)
    family_ids = np.arange(1, family_num + 1)
    families = np.array([family_ids[i % family_num] for i in range(job_count)])

    # 2) family center 랜덤 생성 + 최소 간격 보장
    min_required_gap = max(3, total_range // (family_num + 1))

    centers = []
    tries = 0
    max_tries = 10000

    while len(centers) < family_num and tries < max_tries:
        c = np.random.randint(lo, hi + 1)
        if all(abs(c - x) >= min_required_gap for x in centers):
            centers.append(c)
        tries += 1

    # fallback: 균등 간격
    if len(centers) < family_num:
        step = max(1, total_range // max(1, (family_num - 1)))
        centers = [lo + i * step for i in range(family_num)]
        centers = [min(hi, c) for c in centers]

    family_centers = np.array(centers, dtype=float)  # (family_num,)

    # ─────────────────────────────
    # 🔥 family 안 job 간 편차 크게: 정규분포 사용
    # ─────────────────────────────
    family_centers_for_jobs = family_centers[families - 1]  # (job_count,)

    # job_variation_percent 로 표준편차 결정 (예: 0.5면 전체 range의 50%)
    job_std = max(1.0, total_range * job_variation_percent)

    job_processing_times = np.random.normal(
        loc=family_centers_for_jobs,
        scale=job_std,
        size=job_count,
    )
    job_processing_times = np.clip(job_processing_times, lo, hi).astype(int)  # 범위 밖 잘라줌

    # ─────────────────────────────
    # 5) machine 편차는 작게 (기존과 동일)
    # ─────────────────────────────
    mach_std = np.maximum(1, (job_processing_times * machine_variation_percent))

    processing_base = np.random.normal(
        loc=job_processing_times[:, None],
        scale=mach_std[:, None],
        size=(job_count, machine_count),
    )
    processing_base = np.clip(processing_base, lo, hi).astype(int)

    base_processing_times = pd.DataFrame(
        processing_base, columns=[f"M{i+1}" for i in range(machine_count)]
    )

    # operations_per_job 만큼 반복
    processing_times = pd.DataFrame(
        np.repeat(base_processing_times.values, operations_per_job, axis=0),
        columns=base_processing_times.columns,
    )
    processing_index = [
        f"J{j+1:02}{op:02}"
        for j in range(job_count)
        for op in range(1, operations_per_job + 1)
    ]
    processing_times.index = processing_index

    # 저장
    filename_prefix = (
        f"M{machine_count}_J{job_count}_F{family_type}_"
        f"P{processing_time_range[0]}-{processing_time_range[1]}_"
        f"D{deterioration_rate_range[0]}-{deterioration_rate_range[1]}_"
        f"Fgap_m{machine_variation_percent}_j{job_variation_percent}"
    )
    target_folder = os.path.join(folder_path, filename_prefix)
    os.makedirs(target_folder, exist_ok=True)

    processing_csv_path = os.path.join(
        target_folder, f"{filename_prefix}_U{problem_num}_processing.csv"
    )
    processing_times.to_csv(processing_csv_path)
    print("Processing times CSV saved")

    # 통계
    job_means = base_processing_times.mean(axis=1)
    job_stds = base_processing_times.std(axis=1, ddof=0)

    # ─────────────────────────────
    # Family별 그래프 (y축을 processing_time_range로 고정)
    # ─────────────────────────────
    for fam in range(1, family_num + 1):
        mask = (families == fam)
        if not np.any(mask):
            continue

        job_indices = np.where(mask)[0]
        job_codes = [f"J{j+1:02}" for j in job_indices]

        fam_means = job_means[mask]
        fam_stds = job_stds[mask]

        plt.figure(figsize=(14, 4))
        x = np.arange(len(fam_means))

        plt.errorbar(
            x,
            fam_means,
            yerr=fam_stds,
            fmt="o",
            markersize=4,
            capsize=3,
            label="Job Mean ± Std",
        )

        # family 전체 mean 수평선
        fam_overall_mean = fam_means.mean()
        plt.axhline(
            fam_overall_mean,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"Family Mean = {fam_overall_mean:.1f}",
        )

        plt.xticks(x, job_codes, rotation=90)
        plt.title(f"Family {fam} - Job Processing Time Mean ± Std")
        plt.xlabel("Job")
        plt.ylabel("Processing Time")

        # 🔥 y축을 processing_time_range로 고정
        plt.ylim(lo, hi)

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(
            target_folder, f"{filename_prefix}_U{problem_num}_family{fam}_stats.png"
        )
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Family {fam} stats FIG saved")

    # 열화율 부분은 그대로
    original_deterioration_rates = np.random.uniform(
        deterioration_rate_range[0],
        deterioration_rate_range[1],
        (machine_count, job_count),
    )
    machine_deterioration_rates = 0.01 + ((original_deterioration_rates - 1) / 4) * (
        0.05 - 0.01
    )
    machine_deterioration_rates = np.round(machine_deterioration_rates, 3)

    base_deterioration_rates = pd.DataFrame(
        machine_deterioration_rates.T, columns=[f"M{i+1}" for i in range(machine_count)]
    )
    deterioration_rates = pd.DataFrame(
        np.repeat(base_deterioration_rates.values, operations_per_job, axis=0),
        columns=base_deterioration_rates.columns,
    )
    deterioration_rates.index = processing_index

    det_path = os.path.join(
        target_folder, f"{filename_prefix}_U{problem_num}_deterioration.csv"
    )
    deterioration_rates.to_csv(det_path)
    print("Deterioration rates CSV saved")

    return f"Files saved in folder: {target_folder}"


folder_path = "../PMS_data"
os.makedirs(folder_path, exist_ok=True)

machine_count = 10
job_count = 300
family_type = 3
processing_time_range = (10, 50)
deterioration_rate_range = (5, 10)
operations_per_job = 1
problem_num = 1

# 같은 family 안에서 job 편차는 크게, 하나의 job 안에서 machine 편차는 작게
machine_variation_percent = 0.05   # job 내 machine 간 편차 (작게)
job_variation_percent = 0.5        # family 내 job 간 편차 (크게)

for i in range(1, problem_num + 1):
    msg = generate_processing_times_and_deterioration_rate(
        machine_count,
        job_count,
        family_type,
        processing_time_range,
        deterioration_rate_range,
        operations_per_job,
        folder_path,
        i,
        machine_variation_percent,
        job_variation_percent
    )
    print(msg)