"""The streaming index against a brute-force reference on random text."""

import random

import pytest

from contamscan.methods import Params
from contamscan.scan import EvalItem, scan


def occurs(needle: list[str], hay: list[str]) -> bool:
    n = len(needle)
    return any(hay[j : j + n] == needle for j in range(len(hay) - n + 1))


def lcs_len(a: list[str], b: list[str]) -> int:
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


# six seeds in one process, two through a worker pool (pool start-up is the slow part)
@pytest.mark.parametrize(("seed", "workers"), [(s, 1) for s in range(6)] + [(0, 2), (1, 2)])
def test_index_matches_brute_force(seed, workers):
    rng = random.Random(seed)
    vocab = [f"t{i}" for i in range(6)]  # tiny vocabulary: many chance matches
    items = []
    for i in range(8):
        k = rng.randint(2, 40)
        items.append(EvalItem(f"i{i}", " ".join(rng.choice(vocab) for _ in range(k))))
    docs = []
    for d in range(12):
        words = [rng.choice(vocab) for _ in range(rng.randint(0, 60))]
        if rng.random() < 0.5:  # plant a piece of an item
            src = rng.choice(items).text.split()
            a = rng.randint(0, len(src) - 1)
            b = rng.randint(a, len(src))
            pos = rng.randint(0, len(words))
            words[pos:pos] = src[a:b]
        docs.append((f"d{d}", " ".join(words)))
    params = Params(gpt3_n=5, palm_n=3, llama_n=4, substring_min_tokens=10)
    res = scan(items, docs, params, max_matches=1000, workers=workers)
    dtoks = [d.split() for _, d in docs]
    for it, r in zip(items, res, strict=True):
        t = it.text.split()

        def matched(n, t=t):
            return {p for p in range(len(t) - n + 1) if any(occurs(t[p : p + n], h) for h in dtoks)}

        g = matched(5)
        if len(t) >= 5:
            assert r.gpt3_overlap == pytest.approx(len(g) / (len(t) - 4))
        else:
            whole = any(occurs(t, h) for h in dtoks)
            assert (r.gpt3_label == "dirty") == whole
        pm = matched(3)
        if len(t) >= 3:
            assert r.palm_overlap == pytest.approx(len(pm) / (len(t) - 2))
        else:
            assert r.palm_overlap is None
        cov = set()
        for p in matched(4):
            cov.update(range(p, p + 4))
        assert r.llama2_coverage == pytest.approx(len(cov) / len(t))
        lcs = max(lcs_len(t, h) for h in dtoks)
        # chains of 3-grams find any shared run of 3 or more; a short item can also match whole
        expected = lcs if (lcs >= 3 or (len(t) < 5 and lcs == len(t))) else 0
        assert r.longest_match_tokens == expected


@pytest.mark.parametrize("seed", range(4))
def test_doc_frequency_filter_matches_brute_force(seed):
    rng = random.Random(100 + seed)
    vocab = [f"t{i}" for i in range(4)]
    items = [
        EvalItem(f"i{i}", " ".join(rng.choice(vocab) for _ in range(rng.randint(6, 30))))
        for i in range(6)
    ]
    docs = [
        (f"d{d}", " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 40)))) for d in range(15)
    ]
    k_max = 3
    params = Params(gpt3_n=4, palm_n=3, llama_n=5, max_doc_freq=k_max)
    res = scan(items, docs, params)
    dtoks = [d.split() for _, d in docs]
    for it, r in zip(items, res, strict=True):
        t = it.text.split()

        def kept(n, t=t):
            return {
                p
                for p in range(len(t) - n + 1)
                if 0 < sum(occurs(t[p : p + n], h) for h in dtoks) <= k_max
            }

        assert r.gpt3_overlap == pytest.approx(len(kept(4)) / (len(t) - 3))
        assert r.palm_overlap == pytest.approx(len(kept(3)) / (len(t) - 2))
        cov = set()
        for p in kept(5):
            cov.update(range(p, p + 5))
        assert r.llama2_coverage == pytest.approx(len(cov) / len(t))


def diagonal_runs(t: list[str], h: list[str], min_len: int) -> list[tuple[int, int, int]]:
    """Maximal runs of equal tokens along every diagonal, by direct comparison."""
    runs = []
    for d in range(-len(t), len(h) + 1):
        p = max(0, -d)
        while p < len(t) and p + d < len(h):
            if t[p] != h[p + d]:
                p += 1
                continue
            q = p
            while q < len(t) and q + d < len(h) and t[q] == h[q + d]:
                q += 1
            if q - p >= min_len:
                runs.append((q - p, p, p + d))
            p = q
    return runs


@pytest.mark.parametrize("seed", range(5))
def test_skip_budget_coverage_matches_direct_runs(seed):
    from contamscan.scan import skipgram_cover

    rng = random.Random(200 + seed)
    vocab = [f"t{i}" for i in range(30)]
    items, docs = [], []
    for i in range(5):
        words = [rng.choice(vocab) for _ in range(rng.randint(15, 50))]
        items.append(EvalItem(f"i{i}", " ".join(words)))
        noisy = [w if rng.random() > 0.08 else "zz" for w in words]
        docs.append((f"d{i}", " ".join([rng.choice(vocab) for _ in range(5)] + noisy)))
    params = Params(gpt3_n=13, palm_n=6, llama_n=9, llama_skip_budget=3)
    res = scan(items, docs, params)
    dtoks = [d.split() for _, d in docs]
    for it, r in zip(items, res, strict=True):
        t = it.text.split()
        cov = set()
        for h in dtoks:
            runs = diagonal_runs(t, h, 6)
            for length, p, _ in runs:
                if length >= 9:
                    cov.update(range(p, p + length))
            cov |= skipgram_cover(runs, 3, 9)
        assert r.llama2_coverage == pytest.approx(len(cov) / len(t))


@pytest.mark.parametrize("seed", range(4))
def test_containment_matches_brute_force_with_code_tokens(seed):
    from contamscan.text import normalise

    rng = random.Random(300 + seed)
    vocab = ["ab", "Cd", "e_f", "(", ")", ":", "Gh1"]
    items = [
        EvalItem(f"i{i}", " ".join(rng.choice(vocab) for _ in range(rng.randint(1, 12))))
        for i in range(10)
    ]
    docs = []
    for d in range(10):
        words = [rng.choice(vocab) for _ in range(rng.randint(0, 30))]
        if rng.random() < 0.7:
            planted = rng.choice(items).text
            planted = rng.choice([planted, planted.upper(), planted.swapcase()])
            words.insert(rng.randint(0, len(words)), planted)
        docs.append((f"d{d}", " ".join(words)))
    res = scan(items, docs, Params(gpt3_n=13, palm_n=8, llama_n=11), tokenizer="code")
    for it, r in zip(items, res, strict=True):
        nt = normalise(it.text)
        # the vocabulary keeps whole words apart, so containment is word-aligned here
        expected = bool(nt) and any(
            it.text in d or f" {nt} " in f" {normalise(d)} " for _, d in docs
        )
        assert (r.substring_label == "exact") == expected, it.text
