"""Resource bounds of the per-document scan on repeated tokens."""

import time
import tracemalloc

from contamscan.methods import Params
from contamscan.scan import EvalIndex, EvalItem


def test_repeated_token_document_needs_memory_linear_in_the_item():
    """A separator line repeated through a document hits the same item n-gram at every
    position; the longest-run search must not store every (item, document) pair."""
    item = EvalItem("a", "def f():\n    # " + "=" * 60 + "\n    return 1")
    doc = ("# " + "=" * 79 + "\n") * 250
    index = EvalIndex([item], Params(), "code")
    tracemalloc.start()
    t0 = time.perf_counter()
    hits = index.scan_doc(doc)
    seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert hits.spans[0][0] == 61  # "#" and the 60 "=" tokens
    assert peak < 20 * 2**20, peak
    assert seconds < 20
