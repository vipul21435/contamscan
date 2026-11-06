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
