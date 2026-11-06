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


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("workers", [1, 2])
def test_index_matches_brute_force(seed, workers):
    if workers == 2 and seed > 1:
        pytest.skip("pool start-up cost; two seeds are enough")
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
