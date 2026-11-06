"""Plot the share of items at or above each Llama 2 token-coverage level.

Usage: python examples/plot_coverage.py OUT_DIR TOKENIZER PNG
Reads OUT_DIR/<eval>-<corpus>-<TOKENIZER>-default.json written by run_code_audit.sh.
Needs matplotlib (not a dependency of contamscan).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVALS = {"humaneval": "HumanEval (164)", "mbpp_test": "MBPP test (500)"}
CORPORA = {
    "code_alpaca_20k": ("CodeAlpaca-20k", "#2a78d6"),
    "self_oss_instruct_50k": ("self-oss-instruct 50k", "#eb6834"),
}


def main(out: Path, tok: str, png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    grid = np.linspace(0, 1, 201)
    for ax, (ev, title) in zip(axes, EVALS.items(), strict=True):
        for corpus, (label, colour) in CORPORA.items():
            rep = json.loads((out / f"{ev}-{corpus}-{tok}-default.json").read_text())
            cov = np.array([it["llama2_coverage"] for it in rep["items"]])
            share = [(cov >= g).mean() for g in grid]
            ax.plot(grid, share, color=colour, lw=2, label=label)
        for x, name in ((0.2, "not clean"), (0.8, "dirty")):
            ax.axvline(x, color="#8a8984", lw=1, ls="--")
            ax.text(x + 0.01, 0.93, name, color="#52514e", fontsize=8)
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.set_xlabel("Llama 2 token coverage (11-grams)", fontsize=9, color="#52514e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    axes[0].set_ylabel("share of items at or above", fontsize=9, color="#52514e")
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle(f"Token coverage per item, {tok} tokenizer", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(png, dpi=130, facecolor="#fcfcfb")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
