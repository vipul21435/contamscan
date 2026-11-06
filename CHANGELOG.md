# Changelog

## 0.1.0 (unreleased)

- Word and code tokenisers with character offsets; NFKC, case, punctuation and whitespace
  normalisation.
- Rolling 64-bit n-gram hashes and an in-memory index of the evaluation set.
- Single-pass streaming scan of JSONL, Parquet and plain-text corpora, optionally over
  several processes.
- Methods: GPT-3 any-13-gram, PaLM 70 % of 8-grams, Llama 2 token coverage (clean, mixed,
  dirty) and longest shared span / exact containment; matched documents with character
  offsets.
- Bootstrap interval for the difference between scores on contaminated and clean items.
- `contamscan scan` with JSON and Markdown reports and `--fail-above` for CI.
- Examples: fetch and convert HumanEval, MBPP and CodeAlpaca-20k; seeded synthetic score
  comparison.
