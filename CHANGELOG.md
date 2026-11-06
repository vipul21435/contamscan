# Changelog

## 0.1.0 - 2026-11-06

### Added

- Word and code tokenisers with character offsets; NFKC, case, punctuation and whitespace
  normalisation. Han, hiragana and katakana characters are one token each.
- Rolling 64-bit n-gram hashes and an in-memory index of the evaluation set.
- Single-pass streaming scan of JSONL, gzip JSONL, Parquet and plain-text corpora,
  optionally over several processes. Fields holding lists or objects are read as their
  string values.
- Methods: GPT-3 any-13-gram, PaLM 70 % of 8-grams, Llama 2 token coverage (clean, mixed,
  dirty), and longest shared span with exact whole-item containment; matched documents are
  reported with character offsets.
- Paper variants as flags: `--gpt3-n auto`, `--palm-per-field`, `--max-doc-freq` and
  `--llama-skip-budget` (an approximation of the Llama 2 skipgram budget).
- Score comparison: bootstrap interval for the difference between contaminated and clean
  items, the Llama 2 Z statistic for its four subsets (the finite-population correction is
  available behind `--llama-z-fpc`), and the GPT-3 relative change of the clean-only score.
- `contamscan scan` with JSON and Markdown reports and `--fail-above` for CI. Invalid
  flag values, missing or malformed inputs and a missing field give a one-line error and
  exit status 2, distinct from the status 1 of `--fail-above`.
- Bounded memory in the longest-run search and in the bootstrap.
- Package exports (`EvalItem`, `Params`, `load_eval`, `iter_corpus`, `summarise`, ...) with
  docstrings on the public objects.
- Example scripts: fetch and convert HumanEval, MBPP, CodeAlpaca-20k and self-oss-instruct;
  run and summarise the code audit; plot token coverage; planted-contamination recall;
  seeded synthetic score comparison.
- README with method descriptions, results on public code data and a Limitations section.
- CI with ruff, mypy and pytest on Python 3.10 and 3.13.

### Known limitations

See the Limitations section of the README, including the approximated Llama 2 skipgram
budget, the lack of fuzzy matching and the in-memory index.
