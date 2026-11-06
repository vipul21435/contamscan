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
- Paper variants as flags: `--gpt3-n auto` (5th-percentile length clamped to [8, 13]),
  `--palm-per-field`, `--max-doc-freq` (ignore n-grams common across the corpus) and
  `--llama-skip-budget` (an approximation of the Llama 2 skipgram budget).
- Score comparison: bootstrap interval for the difference between contaminated and clean
  items, the Llama 2 Z statistic for its four subsets, and the GPT-3 relative change of the
  clean-only score.
- `contamscan scan` with JSON and Markdown reports and `--fail-above` for CI.
- Examples: fetch and convert HumanEval, MBPP, CodeAlpaca-20k and self-oss-instruct; run and
  summarise the code audit; plot token coverage; seeded synthetic score comparison.
- Fix: whole-item containment (raw and normalised) is now checked for every item against
  every document through a tokenizer-independent word anchor. Before, it ran only for items
  that already shared an n-gram with the document, and a contained item whose shared span
  was empty was dropped, so re-cased copies were missed with the code tokenizer.
  Containment matches are kept first under `--max-matches`.
- Fix: a `--fields` name missing from every evaluation record, or a corpus with no text in
  any document, now stops with exit status 2 instead of reporting 0 % and passing
  `--fail-above`; partly missing fields and empty documents give warnings.
- Fix: out-of-range flag values (n of 0, thresholds outside [0, 1], `--llama-clean` above
  `--llama-dirty`, `--workers 0`, negative `--max-matches`) and missing input paths are
  usage errors (exit status 2) instead of tracebacks or silent acceptance; a partial
  `--score-field` prints how many items lack a score.
- Fix: the Llama 2 `sigma_n` is now `s / sqrt(n)`, the form the paper's Table 51 is
  consistent with; the finite-population correction used before inflated |Z| of the large
  subsets (clean, not dirty) and is kept behind `--llama-z-fpc`.
- Fix: the longest-run search no longer stores every (item position, document position)
  pair of the chaining n-gram, which took quadratic memory on documents with repeated
  separator lines; runs are now extended while walking the document.
