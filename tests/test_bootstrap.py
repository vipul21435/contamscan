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


def test_subset_z_matches_simulated_sampling_distribution():
    """sigma_n is the spread of means of random n-subsets of the set, and the Z of random
    subsets is close to standard normal (about 5 % beyond |Z| > 2)."""
    from contamscan.bootstrap import subset_z

    rng = np.random.default_rng(5)
    scores = (rng.random(400) < 0.6).astype(float)
    n = 60
    means, zs = [], []
    for _ in range(4000):
        mask = np.zeros(scores.size, dtype=bool)
        mask[rng.choice(scores.size, n, replace=False)] = True
        s = subset_z(scores, mask)
        means.append(s.mean)
        zs.append(s.z)
    s = subset_z(scores, mask)
    assert s.n == n and s.mu == scores.mean()
    assert abs(np.std(means) - s.sigma) < 0.05 * s.sigma
    beyond = np.mean(np.abs(zs) > 2)
    assert 0.03 <= beyond <= 0.07


def test_subset_z_edge_cases_and_relative_change():
    from contamscan.bootstrap import relative_change, subset_z

    assert subset_z([1.0, 0.0], [False, False]) is None
    whole = subset_z([1.0, 0.0, 1.0], [True, True, True])
    assert whole.sigma == 0.0 and whole.z is None
    high = subset_z([1, 1, 0, 0], [True, True, False, False], "dirty")
    assert high.subset == "dirty" and high.z > 0
    assert relative_change([0.5], [0.5, 0.7]) == (0.5 - 0.6) / 0.6
    assert relative_change([], [1.0]) is None
    assert relative_change([1.0], [0.0]) is None
