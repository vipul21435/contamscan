# contamscan

Measure how much of an evaluation set appears in a training corpus, explain each match
with document ids and character offsets, and estimate how much the overlap moves scores.

## Why this is hard

A benchmark item rarely appears in a corpus byte for byte: it is re-indented, re-cased,
split across a prompt and an answer, or quoted in part inside a longer document. Fixed
n-gram rules disagree on the same data (one shared 13-gram makes an item "dirty" under the
GPT-3 rule, while PaLM asks for 70 % of all 8-grams), common code idioms such as
`for i in range(n):` match everywhere, and corpora are far larger than memory. A useful
audit therefore has to stream the corpus once, apply several published rules side by side,
show the matched text so a person can judge it, and check whether contaminated items are
actually scored higher.

## Install

```bash
pip install git+https://github.com/vipul21435/contamscan
# Parquet input needs pyarrow:
pip install "contamscan[parquet] @ git+https://github.com/vipul21435/contamscan"
```

Python 3.10 or newer. The only required dependency is numpy.

## Quickstart

The evaluation set is JSONL with an id, the text fields to check and optionally a per-item
score; the corpus is any mix of JSONL, Parquet and plain-text files or directories.

```bash
contamscan scan eval.jsonl corpus/ --fields question,answer --score-field pass \
    --json report.json --markdown report.md --fail-above 0.05
```

A seeded synthetic run (`python examples/synthetic_comparison.py runs/synthetic`; 200 items
of random words, 60 of them planted whole or in part in 2,060 random documents, simulated
pass rates 0.45 for unplanted and 0.75 for fully planted items; no real model output):

```text
200 items, 2060 documents, 0.5 s
  gpt3           55 contaminated (27.5%)
  palm           40 contaminated (20.0%)
  llama2         40 contaminated (20.0%)
  substring      40 contaminated (20.0%)
```

and the score table from `runs/synthetic/report.md`:

| split | n clean | n contaminated | mean clean | mean contaminated | difference | 95% interval |
|---|---:|---:|---:|---:|---:|---|
| gpt3 | 145 | 55 | 0.469 | 0.782 | +0.313 | [+0.173, +0.445] |
| palm | 160 | 40 | 0.475 | 0.875 | +0.400 | [+0.263, +0.519] |
| llama2 | 160 | 40 | 0.475 | 0.875 | +0.400 | [+0.263, +0.519] |
| substring | 160 | 40 | 0.475 | 0.875 | +0.400 | [+0.263, +0.519] |
| llama2_not_clean_vs_clean | 141 | 59 | 0.461 | 0.780 | +0.319 | [+0.186, +0.448] |

The 40 fully planted items are found by every method. Of the 20 items planted as a third
of their words, 15 are flagged by the any-13-gram rule (55 = 40 + 15) and 19 fall in the
Llama 2 `mixed` band; none reaches the PaLM or Llama 2 `dirty` thresholds.
With `--fail-above F` the command exits with status 1 when the contaminated fraction under
`--fail-method` (default `palm`) is above `F`, so it can gate a CI job.

## How it works

**Tokens.** `--tokenizer word` (default) case-folds, drops punctuation and splits on
whitespace, which follows the GPT-3 rule of comparing words "ignoring all punctuation,
whitespace, and casing". `--tokenizer code` keeps case and emits identifiers, numbers and
each operator character as tokens, because `Foo`/`foo` differ in code and punctuation
carries structure. Every token keeps its character span. The text of an item is its
`--fields` joined by newlines.

**Index.** Each token is hashed with BLAKE2b to 64 bits; an n-gram hash is the polynomial
`sum(t_i * B^(n-1-i)) mod 2^64`, updated in O(1) per position. For each n used by a method
the evaluation set's n-gram hashes go into a dict `hash -> [(item, position)]`. The corpus
is streamed once (`--workers K` spreads documents over K processes); each document's
n-gram hashes are looked up and the hit positions are accumulated per item. Memory grows
with the evaluation set, not the corpus.

**Methods.** For an item with k tokens and g_n = k - n + 1 n-grams:

| method | quantity | label | source of the default |
|---|---|---|---|
| `gpt3` | matched 13-grams / g_13 | `dirty` if any 13-gram is shared; items shorter than 13 tokens are matched whole | Brown et al. 2020, Appendix C ([arXiv 2005.14165](https://arxiv.org/abs/2005.14165)) |
| `palm` | matched 8-grams / g_8 | `contaminated` if >= 0.70 | Chowdhery et al. 2022, Section 8 ([arXiv 2204.02311](https://arxiv.org/abs/2204.02311)) |
| `llama2` | tokens inside any matched 11-gram / k | `clean` < 0.20, `dirty` >= 0.80, `mixed` between | Touvron et al. 2023, Appendix A.6 ([arXiv 2307.09288](https://arxiv.org/abs/2307.09288)) |
| `substring` | longest run of tokens shared with one document | `exact` if the whole item text is contained (raw or normalised), `long-match` if the run is >= 50 tokens | this tool's choice |

GPT-3 picks N per dataset as the 5th-percentile example length, clamped to [8, 13]; here N
is one flag (`--gpt3-n`, default 13). Llama 2 counts a token as contaminated when it lies in
a shared n-gram "longer than 10 tokens", hence the default `--llama-n 11`. The longest run
is found by chaining matches of the smallest n along a diagonal (item position p and
document position j, then p+1 and j+1, ...); a run of r matching n-grams is a shared span of
r + n - 1 tokens. For each item the report keeps the `--max-matches` documents with the
longest spans, with character offsets in both the item and the document.

**Score comparison.** When `--score-field` is set, for every method the items are split into
contaminated and not, and the report gives `mean(contaminated) - mean(clean)` with a
percentile bootstrap interval (`--n-boot`, default 10,000, groups resampled independently,
`--seed`). For Llama 2 it also reports not-clean versus clean, one of the four subsets that
paper compares. A positive difference that excludes zero is evidence that overlap inflates
scores; the GPT-3 and Llama 2 analyses use the same clean/dirty split for that reason.

## Results on public code data

HumanEval (164 items, `prompt` + `canonical_solution`) and the MBPP test split (500 items,
`text` + `code`) scanned against CodeAlpaca-20k (20,022 records, `instruction` + `input` +
`output`). Commands (data in `data/`, fetched with `examples/fetch_code_data.sh` and
converted with `examples/prepare_code_data.py`):

```bash
contamscan scan data/humaneval.jsonl data/code_alpaca_20k.jsonl --fields prompt,solution \
    --corpus-fields instruction,input,output --tokenizer word --workers 4 \
    --json humaneval-word.json --markdown humaneval-word.md
# likewise with mbpp_test.jsonl and with --tokenizer code
```

Items flagged (scan time as printed by the tool; wall time of the whole command in brackets):

| evaluation set | tokenizer | gpt3 dirty | palm >= 70 % | llama2 dirty | substring | items with any shared n-gram | time |
|---|---|---:|---:|---:|---:|---:|---|
| HumanEval | word | 7 (4.3 %) | 0 | 0 | 0 | 55 | 1.2 s (1.9 s) |
| HumanEval | code | 53 (32.3 %) | 0 | 0 | 0 | 139 | 2.0 s (2.1 s) |
| MBPP test | word | 19 (3.8 %) | 0 | 0 | 1 | 163 | 1.6 s (1.8 s) |
| MBPP test | code | 88 (17.6 %) | 1 | 1 | 1 | 289 | 2.1 s (2.3 s) |

Examples from the word-tokenizer runs (item, longest shared run, best document, snippet):

- `mbpp/374`: 52 words shared with `alpaca-18534` (characters 154-450), a string
  permutation routine: `next_list = [] for i in range(0,len(prev_list)): for j in range(0,len(str)): ...`.
  It is the one MBPP item that is `contaminated`/`dirty` under PaLM and Llama 2 with the
  code tokenizer (81.0 % of 8-grams, 86.2 % token coverage).
- `mbpp/226`: the problem statement "Write a python function to remove the characters which
  have odd index values of a given string" is in `alpaca-439` (characters 0-94).
- `mbpp/32`: "Write a python function to find the largest prime factor of a given" is in
  `alpaca-3930` (characters 0-67).
- `HumanEval/18`: 17 words of the canonical solution's sliding-window loop
  `for i in range(len(string) - len(substring) + 1): if string[i:i+len(substring)] == substring`
  are in `alpaca-4222`.
- `HumanEval/94`, `HumanEval/127`: the trial-division primality loop, a common idiom rather
  than a copied problem.

Reading: under the strict GPT-3 rule a few percent of items share a 13-word span with
CodeAlpaca-20k, but most of those spans are standard idioms (prime tests, bubble-sort loops,
number-word dictionaries); no HumanEval item and only one MBPP item reaches the PaLM or
Llama 2 thresholds. The code tokenizer turns operators into tokens, so 13 tokens cover far
less code and the any-13-gram rule fires much more often; with code data its `palm` and
`llama2` columns are the more informative ones.

No per-item model results were used here, so these runs have no score comparison; the
comparison is shown only on the synthetic example above.

Data sources and licences, each checked on its dataset card:

| dataset | licence | page |
|---|---|---|
| HumanEval | MIT | https://huggingface.co/datasets/openai/openai_humaneval |
| MBPP (full, test split) | CC BY 4.0 | https://huggingface.co/datasets/google-research-datasets/mbpp |
| CodeAlpaca-20k | CC BY 4.0 | https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k |

None of this data is stored in the repository.

## Limitations

- The Llama 2 skipgram budget (matched spans may differ in up to four positions) is not
  implemented; matches are exact after tokenisation. Llama 2 also counts model tokens, while
  this tool counts word or code tokens.
- GPT-3's per-dataset choice of N and its rule of ignoring 13-grams that occur in more than
  10 training documents are not implemented; common idioms therefore count as overlap.
- Only the Llama 2 not-clean-versus-clean split is reported; the dirty-versus-not-dirty
  split and the paper's Z statistic are not yet implemented.
- A JSON array corpus file is not supported; convert it to JSONL (see
  `examples/prepare_code_data.py`). Fields of an item are joined, so an n-gram may span two
  fields, whereas PaLM checks question, prompt and target separately.
- No fuzzy or embedding-based matching: paraphrased or translated items are not found.
- The real-data results above have no model scores, so the effect on scores is shown only
  on synthetic data.

## Related tools

- [datasieve](https://github.com/vipul21435/datasieve): drops training records that match a
  reference split during deduplication; contamscan instead audits an evaluation set against
  a corpus, explains each match and measures the effect on scores.
- [evalstats](https://github.com/vipul21435/evalstats): error bars and paired comparisons for
  model evaluations.
- [sftlint](https://github.com/vipul21435/sftlint): lint chat, SFT and tool-calling datasets.
- [taskledger](https://github.com/vipul21435/taskledger): schema validation and lint rules for
  benchmark-task bundles.
- [graderguard](https://github.com/vipul21435/graderguard): audit graders for reward hacking
  and flakiness.

## Licence

MIT, see [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
