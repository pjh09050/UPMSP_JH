import numpy as np
from scipy.stats import mannwhitneyu, t


# ============================================================
# 1. Cmax(makespan) 데이터 입력
#    각 방법당 10개 값을 입력
# ============================================================

data = {
    1: {
        "Ours": [823.487, 798.358, 785.883, 823.561, 826.994, 775.388, 769.942, 759.357, 805.119, 822.801],
        "RNN-PPO": [2067.328, 2524.704, 2195.296, 1913.297, 1828.016, 2096.652, 2545.087, 2565.901, 2130.17, 1749.942],
        "Flexible-GA": [2067.328, 2524.704, 2195.296, 1913.297, 1828.016, 2003.868, 2274.830, 2289.269, 2151.665, 2157.351],
        "Hybrid-GA": [2496.139, 2314.368, 2377.717, 2409.623, 2419.172, 2227.013, 2257.681, 2628.908, 2702.742, 2107.67],
        "WOA-VNS": [2151.189, 2103.546, 2383.858, 2126.828, 2238.384, 2346.225, 2334.707, 2274.307, 2299.551, 2111.535],
    },

    12: {
        "Ours": [2488.465, 2466.776, 2455.735, 2492.094, 2514.112, 2417.425, 2460.645, 2508.362, 2441.896, 2508.412],
        "RNN-PPO": [11280.273, 17353.444, 9870.772, 7157.164, 19921.993, 8045.528, 17614.209, 21380.382, 7654.359, 5319.241],
        "Flexible-GA": [6748.655, 6763.445, 6693.854, 6841.601, 6718.741, 6721.029, 6735.597, 6657.383, 6766.338, 6635.210],
        "Hybrid-GA": [13370.682, 13331.206, 13504.678, 12227.872, 13001.406, 12808.441, 12298.096, 13437.375, 12787.098, 14100.462],
        "WOA-VNS": [5708.487, 5827.206, 6669.164, 5706.952, 6270.573, 5840.854, 5580.95, 5844.589, 5548.008, 5874.495],
    },

    13: {
        "Ours": [968.237, 955.943, 935.23, 945.624, 973.783, 902.524, 920.827, 886.756, 838.529, 945.81],
        "RNN-PPO": [2424.17, 2787.608, 2112.163, 2398.005, 2239.408, 2933.48, 2845.45, 2835.16, 2763.467, 2540.083],
        "Flexible-GA": [2982.040, 2898.604, 2967.606, 2834.177, 2900.091, 2833.423, 2981.571, 2795.54, 2810.881, 2693.720],
        "Hybrid-GA": [2996.603, 3126.49, 3133.055, 3100.067, 3051.583, 3106.179, 3239.391, 3060.329, 3167.968, 3146.434],
        "WOA-VNS": [2817.709, 2654.558, 2939.490, 3004.245, 2764.403, 2687.68, 2770.326, 2816.107, 2790.952, 2900.067],
    },

    20: {
        "Ours": [2586.529, 2796.673, 2697.074, 2571.572, 2751.085, 2683.727, 2746.642, 2575.497, 2658.145, 2583.93],
        "RNN-PPO": [20531.689, 21791.45, 9382.839, 15235.461, 21505.423, 22217.674, 9416.497, 18301.909, 8465.443, 9590.8],
        "Flexible-GA": [7018.066, 7438.54, 7285.758, 7128.348, 7066.946, 7229.669, 7125.937, 7285.688, 7175.089, 7298.428],
        "Hybrid-GA": [13694.356, 11720.951, 12907.544, 12683.759, 12315.902, 14709.186, 14005.115, 14021.159, 14574.393, 16113.634],
        "WOA-VNS": [6577.488, 6789.432, 7240.334, 6864.046, 6984.113, 6983.928, 7028.447, 6568.063, 7036.365, 7054.508],
    },
}

# ============================================================
# 2. Cliff's Delta
#
#    Cmax는 작을수록 좋음
#
#    delta > 0 : Ours가 좋음
#    delta = 0 : 비슷함
#    delta < 0 : Baseline이 좋음
# ============================================================

def cliffs_delta(ours, baseline):

    ours = np.asarray(ours, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    wins = 0
    losses = 0

    for o in ours:
        for b in baseline:

            # Ours의 Cmax가 더 작음 → Ours 승리
            if o < b:
                wins += 1

            # Baseline의 Cmax가 더 작음 → Baseline 승리
            elif o > b:
                losses += 1

            # o == b 는 tie → 0으로 처리

    total_pairs = len(ours) * len(baseline)

    delta = (wins - losses) / total_pairs

    return delta


# ============================================================
# 3. Cliff's Delta 크기 해석
# ============================================================

def cliff_magnitude(delta):

    abs_delta = abs(delta)

    if abs_delta < 0.147:
        return "negligible"

    elif abs_delta < 0.33:
        return "small"

    elif abs_delta < 0.474:
        return "medium"

    else:
        return "large"


# ============================================================
# 4. Welch 기반 ΔCmax의 95% CI
#
# ΔCmax = mean(Baseline) - mean(Ours)
#
# 양수 → Ours가 좋음
# ============================================================

def welch_ci(ours, baseline, confidence=0.95):

    ours = np.asarray(ours, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    n_o = len(ours)
    n_b = len(baseline)

    mean_o = np.mean(ours)
    mean_b = np.mean(baseline)

    # sample variance (ddof=1)
    var_o = np.var(ours, ddof=1)
    var_b = np.var(baseline, ddof=1)

    # 평균 차이
    delta = mean_b - mean_o

    # Standard Error
    term_b = var_b / n_b
    term_o = var_o / n_o

    se = np.sqrt(term_b + term_o)

    # Welch-Satterthwaite degrees of freedom
    numerator = (term_b + term_o) ** 2

    denominator = (
        (term_b ** 2) / (n_b - 1)
        +
        (term_o ** 2) / (n_o - 1)
    )

    df = numerator / denominator

    # 95% CI → alpha = 0.05
    alpha = 1 - confidence

    t_critical = t.ppf(
        1 - alpha / 2,
        df
    )

    margin = t_critical * se

    ci_low = delta - margin
    ci_high = delta + margin

    return delta, ci_low, ci_high, df


# ============================================================
# 5. Mann-Whitney U
#
# H0 : Ours와 Baseline의 분포 차이가 없음
# H1 : Ours의 Cmax가 Baseline보다 작음
#
# 즉, Ours가 더 좋다는 one-sided test
# ============================================================

def mann_whitney_test(ours, baseline):

    ours = np.asarray(ours, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    combined = np.concatenate([ours, baseline])

    # tie가 있는지 확인
    has_ties = len(np.unique(combined)) < len(combined)

    # Tie가 없으면 exact test
    # Tie가 있으면 asymptotic 사용
    if not has_ties:

        method = "exact"

    else:

        method = "asymptotic"

    u_stat, p_value = mannwhitneyu(
        ours,
        baseline,
        alternative="less",
        method=method
    )

    return u_stat, p_value, method


# ============================================================
# 6. 모든 통계량 계산
# ============================================================

def calculate_statistics(ours, baseline):

    ours = np.asarray(ours, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    # --------------------------
    # Mean
    # --------------------------

    ours_mean = np.mean(ours)
    baseline_mean = np.mean(baseline)

    # --------------------------
    # Sample SD
    # ddof=1
    # --------------------------

    ours_sd = np.std(
        ours,
        ddof=1
    )

    baseline_sd = np.std(
        baseline,
        ddof=1
    )

    # --------------------------
    # ΔCmax + Welch 95% CI
    # --------------------------

    delta, ci_low, ci_high, df = welch_ci(
        ours,
        baseline
    )

    # --------------------------
    # Cliff's Delta
    # --------------------------

    cliff = cliffs_delta(
        ours,
        baseline
    )

    magnitude = cliff_magnitude(
        cliff
    )

    # --------------------------
    # Mann-Whitney U
    # --------------------------

    u_stat, p_value, mw_method = mann_whitney_test(
        ours,
        baseline
    )

    return {
        "ours_mean": ours_mean,
        "ours_sd": ours_sd,

        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,

        "delta": delta,

        "ci_low": ci_low,
        "ci_high": ci_high,

        "welch_df": df,

        "cliff": cliff,
        "cliff_magnitude": magnitude,

        "u_stat": u_stat,
        "p_value": p_value,

        "mw_method": mw_method,
    }


# ============================================================
# 7. 결과 출력
# ============================================================

print()
print("=" * 145)

cliff_header = "Cliff's δ"

print(
    f"{'Instance':<10}"
    f"{'Baseline':<16}"
    f"{'Ours mean ± SD':<22}"
    f"{'Baseline mean ± SD':<25}"
    f"{'ΔCmax [95% CI]':<32}"
    f"{cliff_header:<14}"
    f"{'Effect':<13}"
    f"{'p-value':<15}"
)

print("=" * 145)


for instance, methods in data.items():

    # 데이터가 비어 있는 instance는 건너뜀
    if len(methods["Ours"]) == 0:
        continue

    ours = methods["Ours"]

    for baseline_name, baseline in methods.items():

        if baseline_name == "Ours":
            continue

        # 데이터가 비어 있으면 건너뜀
        if len(baseline) == 0:
            continue

        result = calculate_statistics(
            ours,
            baseline
        )

        ours_text = (
            f"{result['ours_mean']:.2f} "
            f"± {result['ours_sd']:.2f}"
        )

        baseline_text = (
            f"{result['baseline_mean']:.2f} "
            f"± {result['baseline_sd']:.2f}"
        )

        delta_ci_text = (
            f"{result['delta']:.2f} "
            f"[{result['ci_low']:.2f}, "
            f"{result['ci_high']:.2f}]"
        )

        print(
            f"{instance:<10}"
            f"{baseline_name:<16}"
            f"{ours_text:<22}"
            f"{baseline_text:<25}"
            f"{delta_ci_text:<32}"
            f"{result['cliff']:<14.3f}"
            f"{result['cliff_magnitude']:<13}"
            f"{result['p_value']:<15.6g}"
        )


print("=" * 145)