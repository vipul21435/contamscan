import numpy as np

from contamscan.bootstrap import compare


def test_empty_group_gives_none():
    assert compare([], [1.0]) is None
    assert compare([1.0], []) is None


def test_point_estimate_and_determinism():
    c = compare([0, 0, 1, 1], [1, 1, 1, 0], seed=3)
    assert c.difference == 0.75 - 0.5
    assert c.low <= c.difference <= c.high
    assert c == compare([0, 0, 1, 1], [1, 1, 1, 0], seed=3)


def test_interval_coverage_by_simulation():
    """Over repeated draws with a known true difference, the 90 % interval covers it
    close to 90 % of the time (binary pass/fail scores, as for pass@1)."""
    rng = np.random.default_rng(7)
    p_clean, p_dirty, truth = 0.4, 0.6, 0.2
    reps, hits = 300, 0
    for r in range(reps):
        a = rng.random(120) < p_clean
        b = rng.random(80) < p_dirty
        c = compare(a, b, n_boot=1000, confidence=0.90, seed=r)
        hits += c.low <= truth <= c.high
    assert 0.84 <= hits / reps <= 0.96


def test_interval_width_matches_normal_theory():
    rng = np.random.default_rng(11)
    a = rng.normal(0, 1, 400)
    b = rng.normal(0.5, 1, 400)
    c = compare(a, b, n_boot=4000)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert abs((c.high - c.low) - 2 * 1.96 * se) < 0.25 * 2 * 1.96 * se


def _simulated_sigma_and_z(fpc: bool):
    from contamscan.bootstrap import subset_z

    rng = np.random.default_rng(5)
    scores = (rng.random(400) < 0.6).astype(float)
    n = 60
    means, zs = [], []
    for _ in range(4000):
        if fpc:  # an n-subset of the set
            idx = rng.choice(scores.size, n, replace=False)
            mask = np.zeros(scores.size, dtype=bool)
            mask[idx] = True
            s = subset_z(scores, mask, fpc=True)
            means.append(s.mean)
            zs.append(s.z)
        else:  # n draws with replacement: the sampling distribution of the mean
            x = scores[rng.integers(0, scores.size, n)]
            means.append(x.mean())
            zs.append((x.mean() - scores.mean()) / (scores.std() / np.sqrt(n)))
    mask = np.zeros(scores.size, dtype=bool)
    mask[:n] = True
    s = subset_z(scores, mask, fpc=fpc)
    assert s.n == n and s.mu == scores.mean()
    return s, means, zs


def test_subset_z_sigma_is_the_spread_of_the_mean_of_n_draws():
    """By default sigma_n = s / sqrt(n), the spread of the mean of n independent draws."""
    for fpc in (False, True):
        s, means, zs = _simulated_sigma_and_z(fpc)
        assert abs(np.std(means) - s.sigma) < 0.05 * s.sigma
        beyond = np.mean(np.abs(zs) > 2)
        assert 0.03 <= beyond <= 0.07


def test_subset_z_complement_ratio_matches_llama2_table_51():
    """Llama 2 Table 51 (70B): HellaSwag clean n=7391 Z=-5.73, not clean n=2651 Z=9.56;
    MMLU-Overall clean n=11862 Z=-2.00, not clean n=2180 Z=4.64. For a subset and its
    complement |Z_small| / |Z_large| = sqrt(n_large / n_small) holds only for
    sigma_n = s / sqrt(n); the finite-population form gives equal and opposite Z."""
    from contamscan.bootstrap import subset_z

    assert abs(9.56 / 5.73 - np.sqrt(7391 / 2651)) < 0.01
    assert abs(4.64 / 2.00 - np.sqrt(11862 / 2180)) < 0.02
    rng = np.random.default_rng(1)
    scores = (rng.random(1000) < 0.6).astype(float)
    mask = np.zeros(1000, dtype=bool)
    mask[:260] = True
    scores[:260] = (rng.random(260) < 0.8).astype(float)
    small, large = subset_z(scores, mask), subset_z(scores, ~mask)
    assert abs(abs(small.z) / abs(large.z) - np.sqrt(740 / 260)) < 1e-9
    assert small.sigma == np.sqrt(scores.var() / 260)
    f_small, f_large = subset_z(scores, mask, fpc=True), subset_z(scores, ~mask, fpc=True)
    assert abs(f_small.z + f_large.z) < 1e-9


def test_subset_z_edge_cases_and_relative_change():
    from contamscan.bootstrap import relative_change, subset_z

    assert subset_z([1.0, 0.0], [False, False]) is None
    whole = subset_z([1.0, 0.0, 1.0], [True, True, True])
    assert whole.sigma > 0.0 and whole.z == 0.0
    whole = subset_z([1.0, 0.0, 1.0], [True, True, True], fpc=True)
    assert whole.sigma == 0.0 and whole.z is None
    assert subset_z([1.0, 1.0], [True, False]).z is None
    high = subset_z([1, 1, 0, 0], [True, True, False, False], "dirty")
    assert high.subset == "dirty" and high.z > 0
    assert relative_change([0.5], [0.5, 0.7]) == (0.5 - 0.6) / 0.6
    assert relative_change([], [1.0]) is None
    assert relative_change([1.0], [0.0]) is None
