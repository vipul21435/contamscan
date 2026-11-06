"""contamscan: measure benchmark contamination of an evaluation set in a training corpus."""

from .io import iter_corpus, load_eval
from .methods import Params
from .report import comparisons, summarise
from .scan import EvalItem, ItemResult, Match

# The scan function is not re-exported here: the name would shadow the ``contamscan.scan``
# module. Import it as ``from contamscan.scan import scan``.

__version__ = "0.1.0"

__all__ = [
    "EvalItem",
    "ItemResult",
    "Match",
    "Params",
    "comparisons",
    "iter_corpus",
    "load_eval",
    "summarise",
]
