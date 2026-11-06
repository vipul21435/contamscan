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

The evaluation set is JSONL (or Parquet) with an id, the text fields to check and
optionally a per-item score; the corpus is any mix of JSONL, Parquet and plain-text files
or directories, read one record at a time.

```bash
contamscan scan eval.jsonl corpus/ --fields question,answer --score-field pass \
    --json report.json --markdown report.md --fail-above 0.05
```

A real run, HumanEval against the 50,661 records of a public code SFT set (see
[Results](#results-on-public-code-data) for the data and its licence):

```text
$ contamscan scan data/jsonl/humaneval.jsonl data/raw/self_oss_instruct_50k.parquet \
    --fields prompt,solution --corpus-fields instruction,response --tokenizer code --workers 4
164 items, 50661 documents, 7.6 s
  gpt3           80 contaminated (48.8%)
  palm            0 contaminated (0.0%)
  llama2          0 contaminated (0.0%)
  substring       4 contaminated (2.4%)
```

With `--fail-above F` the command exits with status 1 when the contaminated fraction under
`--fail-method` (default `palm`) is above `F`, so it can gate a CI job.

The score comparison needs per-item scores. A seeded synthetic run shows it
(`python examples/synthetic_comparison.py runs/synthetic`, 0.7 s wall time): 200 items of
random words, 60 of them planted whole or in part in 2,060 random documents, simulated
pass rates 0.45 for unplanted and 0.75 for fully planted items. **These scores are
simulated, not model output.**

```text
200 items, 2060 documents, 0.5 s
  gpt3           55 contaminated (27.5%)
  palm           40 contaminated (20.0%)
  llama2         40 contaminated (20.0%)
  substring      40 contaminated (20.0%)
```

From `runs/synthetic/report.md` (the `llama2` row is dirty versus not dirty):

| split | n clean | n contaminated | mean clean | mean contaminated | difference | 95% interval |
|---|---:|---:|---:|---:|---:|---|
| gpt3 | 145 | 55 | 0.469 | 0.782 | +0.313 | [+0.173, +0.445] |
| palm | 160 | 40 | 0.475 | 0.875 | +0.400 | [+0.263, +0.519] |
| llama2 | 160 | 40 | 0.475 | 0.875 | +0.400 | [+0.263, +0.519] |
| substring | 160 | 40 | 0.475 | 0.875 | +0.400 | [+0.263, +0.519] |
| llama2_not_clean_vs_clean | 141 | 59 | 0.461 | 0.780 | +0.319 | [+0.186, +0.448] |

| Llama 2 subset | n | mean | mu_n | sigma_n | Z |
|---|---:|---:|---:|---:|---:|
| clean | 141 | 0.461 | 0.555 | 0.023 | -4.13 |
| not clean | 59 | 0.780 | 0.555 | 0.054 | +4.13 |
| not dirty | 160 | 0.475 | 0.555 | 0.018 | -4.54 |
| dirty | 40 | 0.875 | 0.555 | 0.070 | +4.54 |

GPT-3 clean-only score 0.469 on 145 items against 0.555 on all 200: relative change
-15.5 %.

The 40 fully planted items are found by every method. Of the 20 items planted as a third
of their words, 15 are flagged by the any-13-gram rule (55 = 40 + 15) and 19 fall in the
Llama 2 band between clean and dirty; none reaches the PaLM or Llama 2 dirty thresholds.
All four Llama 2 subsets have |Z| > 2, the paper's condition for calling a set affected.

## How it works

**Tokens.** `--tokenizer word` (default) case-folds, drops punctuation and splits on
whitespace, which follows the GPT-3 rule of comparing words "ignoring all punctuation,
whitespace, and casing". `--tokenizer code` keeps case and emits identifiers, numbers and
each operator character as tokens, because `Foo`/`foo` differ in code and punctuation
carries structure. Every token keeps its character span. The text of an item is its
`--fields` joined by newlines. Exact containment is checked on the raw text and on a
normalised form (NFKC, case-folded, punctuation removed, whitespace collapsed).

**Index.** Each token is hashed with BLAKE2b to 64 bits; an n-gram hash is the polynomial
`sum(t_i * B^(n-1-i)) mod 2^64`, updated in O(1) per position. For each n used by a method
the evaluation set's n-gram hashes go into a dict `hash -> [(item, position)]`. The corpus
is streamed once (`--workers K` spreads documents over K processes); each document's
n-gram hashes are looked up and the hit positions are accumulated per item. Memory grows
with the evaluation set, not the corpus. With 64-bit hashes the chance of a false match
between two given n-grams is about 2^-64. A property test checks the index against a
brute-force scan of random text with a four- to six-word vocabulary.

**Methods.** For an item with k tokens and g_n = k - n + 1 n-grams:

| method | quantity | label | source of the default |
|---|---|---|---|
| `gpt3` | matched 13-grams / g_13 | `dirty` if any 13-gram is shared; items shorter than 13 tokens are matched whole | Brown et al. 2020, Appendix C ([arXiv 2005.14165](https://arxiv.org/abs/2005.14165)) |
| `palm` | matched 8-grams / g_8 | `contaminated` if >= 0.70 | Chowdhery et al. 2022, Section 8 ([arXiv 2204.02311](https://arxiv.org/abs/2204.02311)) |
| `llama2` | tokens inside any matched 11-gram / k | `clean` < 0.20, `dirty` >= 0.80, `mixed` between | Touvron et al. 2023, Appendix A.6 ([arXiv 2307.09288](https://arxiv.org/abs/2307.09288)) |
| `substring` | longest run of tokens shared with one document | `exact` if the whole item text is contained (raw or normalised), `long-match` if the run is >= 50 tokens | this tool's choice |

Flags and where their defaults come from:

- `--gpt3-n 13`: GPT-3 uses the 5th-percentile example length in words, with "a minimum
  value of 8" and "a maximum value of 13". `--gpt3-n auto` applies that rule to the
  evaluation set (lower order statistic, clamped to [8, 13]).
- `--palm-n 8 --palm-threshold 0.70`: PaLM counts an example as contaminated if "at least
  70% of the 8-grams in question, prompt, or target were seen at least once". By default
  the fields are joined; `--palm-per-field` takes the fraction over the n-grams inside each
  field and flags the item when any field reaches the threshold.
- `--llama-n 11 --llama-clean 0.20 --llama-dirty 0.80`: Llama 2 counts a token as
  contaminated if it lies in a shared n-gram "longer than 10 tokens"; clean is below 20 %,
  dirty is at or above 80 %. The paper repeats the analysis with minimum match lengths
  10 to 50; pass a larger `--llama-n` to do the same.
- `--max-doc-freq K` (off by default): ignore item n-grams found in more than K documents
  for `gpt3`, `palm` and `llama2`. GPT-3 "ignored 13-grams that matched more than 10
  training documents" in its training-set filter because they were mostly common phrases
  and boilerplate; for code, `K = 10` drops stock idioms.
- `--substring-min-tokens 50`: not from a paper; a run of 50 tokens is long enough that
  common code idioms rarely reach it (see the results below).

The longest run is found by chaining matches of the smallest n along a diagonal (item
position p and document position j, then p+1 and j+1, ...); a run of r matching n-grams is a
shared span of r + n - 1 tokens. For each item the report keeps the `--max-matches`
documents with the longest spans, with character offsets in both the item and the document.

**Score comparison.** When `--score-field` is set the report has three parts.

1. For every method, `mean(contaminated) - mean(clean)` with a percentile bootstrap
   interval (`--n-boot`, default 10,000; the two groups resampled independently; `--seed`).
   For Llama 2 both of the paper's splits are given: dirty versus not dirty, and not clean
   versus clean. A test checks the interval's coverage by simulation.
2. The Llama 2 statistic for each of the four subsets (clean, not clean, not dirty,
   dirty): `Z_n = (mean(subset) - mu_n) / sigma_n`, where `mu_n` is the mean over all N items
   and `sigma_n^2 = s^2 / n * (N - n) / (N - 1)` is the variance of the mean of n items
   drawn without replacement from the set (`s^2` the population variance). The paper calls
   a dataset affected when all four subsets have |Z_n| > 2. It does not say whether its
   sampling is with or without replacement; this tool draws without replacement, which
   makes the Z of a subset and of its complement equal in size and opposite in sign. A
   test checks `sigma_n` against the spread of means of random subsets.
3. GPT-3's comparison: the score on the clean-only items against the score on all items,
   as a relative change `(clean - full) / full`. GPT-3 reads a clean score "more than 1% or
   2% worse than the overall score" as a sign of overfitting to seen examples.

## Results on public code data

HumanEval (164 items, `prompt` + `canonical_solution`) and the MBPP test split (500 items,
`text` + `code`) scanned against two public code instruction sets: CodeAlpaca-20k
(20,022 records, `instruction` + `input` + `output`) and
bigcode/self-oss-instruct-sc2-exec-filter-50k (50,661 records, `instruction` +
`response`, read directly from Parquet). Each pair is scanned with both tokenisers, once
with default flags and once with the paper variants
`--gpt3-n auto --palm-per-field --max-doc-freq 10`. Commands:

```bash
bash examples/fetch_code_data.sh data/raw                 # about 98 MB
python examples/prepare_code_data.py data/raw data/jsonl  # Parquet/JSON -> JSONL
bash examples/run_code_audit.sh data runs 4               # 16 scans, 4 worker processes
python examples/summarise_runs.py runs
```

The 16 scans took 56 s of wall time in total on an 8-core laptop. Items flagged (N is the
GPT-3 n-gram size used; `auto` chose 13 everywhere because the 5th-percentile item is
longer than 13 tokens; "any overlap" counts items sharing at least one n-gram of any
size used):

| eval | corpus | tokenizer | flags | N | gpt3 dirty | palm | llama2 dirty | llama2 not clean | substring | any overlap | scan s | wall s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| humaneval | code_alpaca_20k | word | default | 13 | 7 (4.3 %) | 0 | 0 | 3 (1.8 %) | 0 | 55 | 0.8 | 0.9 |
| humaneval | code_alpaca_20k | word | paper | 13 | 7 (4.3 %) | 1 (0.6 %) | 0 | 3 (1.8 %) | 0 | 55 | 0.8 | 0.9 |
| humaneval | code_alpaca_20k | code | default | 13 | 53 (32.3 %) | 0 | 0 | 18 (11.0 %) | 0 | 139 | 1.2 | 1.3 |
| humaneval | code_alpaca_20k | code | paper | 13 | 48 (29.3 %) | 1 (0.6 %) | 0 | 16 (9.8 %) | 0 | 139 | 1.1 | 1.3 |
| humaneval | self_oss_instruct_50k | word | default | 13 | 15 (9.1 %) | 0 | 0 | 8 (4.9 %) | 0 | 94 | 4.8 | 4.9 |
| humaneval | self_oss_instruct_50k | word | paper | 13 | 14 (8.5 %) | 2 (1.2 %) | 0 | 8 (4.9 %) | 0 | 94 | 4.3 | 4.4 |
| humaneval | self_oss_instruct_50k | code | default | 13 | 80 (48.8 %) | 0 | 0 | 52 (31.7 %) | 4 (2.4 %) | 160 | 7.6 | 7.7 |
| humaneval | self_oss_instruct_50k | code | paper | 13 | 76 (46.3 %) | 2 (1.2 %) | 0 | 39 (23.8 %) | 4 (2.4 %) | 160 | 6.9 | 7.0 |
| mbpp_test | code_alpaca_20k | word | default | 13 | 19 (3.8 %) | 0 | 0 | 28 (5.6 %) | 1 (0.2 %) | 163 | 0.8 | 0.9 |
| mbpp_test | code_alpaca_20k | word | paper | 13 | 19 (3.8 %) | 3 (0.6 %) | 0 | 28 (5.6 %) | 1 (0.2 %) | 163 | 0.8 | 0.9 |
| mbpp_test | code_alpaca_20k | code | default | 13 | 88 (17.6 %) | 1 (0.2 %) | 1 (0.2 %) | 83 (16.6 %) | 1 (0.2 %) | 289 | 1.1 | 1.2 |
| mbpp_test | code_alpaca_20k | code | paper | 13 | 85 (17.0 %) | 4 (0.8 %) | 1 (0.2 %) | 79 (15.8 %) | 1 (0.2 %) | 289 | 1.2 | 1.3 |
| mbpp_test | self_oss_instruct_50k | word | default | 13 | 12 (2.4 %) | 0 | 0 | 25 (5.0 %) | 0 | 202 | 4.4 | 4.5 |
| mbpp_test | self_oss_instruct_50k | word | paper | 13 | 12 (2.4 %) | 5 (1.0 %) | 0 | 25 (5.0 %) | 0 | 202 | 4.6 | 4.7 |
| mbpp_test | self_oss_instruct_50k | code | default | 13 | 105 (21.0 %) | 0 | 0 | 108 (21.6 %) | 0 | 330 | 7.0 | 7.1 |
| mbpp_test | self_oss_instruct_50k | code | paper | 13 | 100 (20.0 %) | 0 | 0 | 96 (19.2 %) | 0 | 330 | 6.6 | 6.8 |

`--workers` on the largest scan (HumanEval against self-oss-instruct, code tokenizer):
19.8 s with 1 process, 6.7 s with 4, 6.2 s with 8 (scan time as printed by the tool).

![Share of items at or above each Llama 2 token-coverage level](docs/coverage-code.png)

`docs/coverage-code.png` (`python examples/plot_coverage.py runs code docs/coverage-code.png`,
needs matplotlib) shows, for the default code-tokenizer runs, the share of items whose
token coverage is at or above each level. No HumanEval item and one MBPP item reach the
dirty line.

Examples (item, document id in the corpus, character offsets in that document):

- `mbpp/374` against CodeAlpaca-20k, `alpaca-18534` characters 153-450: a string
  permutation routine, `next_list = [] for i in range(0,len(prev_list)): for j in
  range(0,len(str)): ...`. Code tokenizer: 86 shared tokens, 81.0 % of 8-grams, 86.2 %
  token coverage, so it is `contaminated` under PaLM and the only item `dirty` under Llama 2
  in any run.
- `HumanEval/19` against self-oss-instruct, document `141434` characters 1391-1503: the
  number-word dictionary `'zero': 0, 'one': 1, ... 'nine': 9` (59 code tokens). Likewise
  `HumanEval/105` (`1: "One", 2: "Two", ...`, document `336305`), `HumanEval/156` (the
  Roman-numeral symbol list, document `252334`) and `HumanEval/96` (a trial-division prime
  loop, document `158426`). These are the four `long-match` items; each is a shared
  table or idiom, not a copied problem, and each item's token coverage stays below 47 %.
- `HumanEval/55` against self-oss-instruct, document `79440` characters 144-237: the
  recursive Fibonacci body `if n == 0: return 0 if n == 1: return 1 return fib(n - 1) +
  fib(n - 2)`. With `--palm-per-field` the solution field is 100 % matched and the item is
  flagged; joined with the prompt it is at 35.7 %.
- `mbpp/226` against CodeAlpaca-20k, `alpaca-439` characters 0-94: the problem statement
  "Write a python function to remove the characters which have odd index values of a given
  string".
- `mbpp/90` against self-oss-instruct, document `5760` characters 42-99: "Write a python
  function to find the length of the longest", flagged only by `--palm-per-field` because
  a one-sentence MBPP statement has few 8-grams and a stock opening fills most of them.

Reading: under the strict GPT-3 rule a few percent of items share a 13-word span with
either corpus, and with the code tokenizer (where 13 tokens cover much less code) a fifth to
a half do; most of those spans are idioms such as prime tests, sorting loops and
number-word tables. No HumanEval item and at most one MBPP item reaches the PaLM or Llama 2
thresholds with joined fields. Per-field PaLM adds a handful of items, mostly short MBPP
statements whose wording is a template ("Write a python function to find the ..."), so
its flags need reading before they are believed. `--max-doc-freq 10` removes 0 to 5 GPT-3
flags per run and lowers the not-clean count by up to 13 items (HumanEval, code tokenizer,
self-oss-instruct), which is the expected effect of dropping n-grams that occur in many
documents.

No per-item model results were used here, so these runs have no score comparison; it is
shown only on the synthetic example above. EvalPlus publishes greedy model samples for
HumanEval on its GitHub releases page, but not per-problem pass/fail verdicts, and
producing verdicts would mean executing model-generated code, which was not done.

Data sources and licences, each checked on its dataset card:

| dataset | licence | page |
|---|---|---|
| HumanEval | MIT | https://huggingface.co/datasets/openai/openai_humaneval |
| MBPP (full, test split) | CC BY 4.0 | https://huggingface.co/datasets/google-research-datasets/mbpp |
| CodeAlpaca-20k | CC BY 4.0 | https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k |
| self-oss-instruct-sc2-exec-filter-50k | ODC-By | https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k |

None of this data is stored in the repository. iamtarun/python_code_instructions_18k_alpaca
was considered and not used because its card states no licence.

## Limitations

- The Llama 2 skipgram budget (matched spans may differ in up to four positions) is not
  implemented; matches are exact after tokenisation. Llama 2 also counts model tokens,
  while this tool counts word or code tokens.
- `--max-doc-freq` does not apply to the longest-span (`substring`) method or to the
  whole-item match of items shorter than N.
- The Llama 2 sampling distribution is computed analytically for sampling without
  replacement; the paper does not state its choice, so Z values may differ from a
  with-replacement calculation by the factor `sqrt((N - n) / (N - 1))`.
- A JSON array corpus file is not supported; convert it to JSONL (see
  `examples/prepare_code_data.py`). With joined fields an n-gram may span two fields;
  `--palm-per-field` changes only the PaLM fraction.
- No fuzzy or embedding-based matching: paraphrased, renamed or translated items are not
  found.
- The real-data results above have no model scores, so the effect on scores is shown only
  on synthetic data.
- The index is an in-memory dict, fine for benchmarks of thousands of items; an evaluation
  set of many millions of n-grams would need a more compact structure.

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
