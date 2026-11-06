# Code audit: all 24 scans

HumanEval (164 items) and the MBPP test split (500 items) against CodeAlpaca-20k (20,022
records) and bigcode/self-oss-instruct-sc2-exec-filter-50k (50,661 records), with both
tokenisers, in three settings: default flags (`default`), the GPT-3 and PaLM variants
`--gpt3-n auto --palm-per-field --max-doc-freq 10` (`paper`), and the approximate Llama 2
skipgram budget `--llama-skip-budget 4` (`skip4`). Data sources and licences are in the
README. Commands:

```bash
bash examples/fetch_code_data.sh data/raw                 # about 98 MB
python examples/prepare_code_data.py data/raw data/jsonl  # Parquet/JSON -> JSONL
bash examples/run_code_audit.sh data runs 4               # 24 scans, 4 worker processes
python examples/summarise_runs.py runs
```

N is the GPT-3 n-gram size used; `auto` chose 13 everywhere because the 5th-percentile item
is longer than 13 tokens. "any overlap" counts items sharing at least one n-gram of any
size used with some document, or contained whole in one.

The counts below were produced again at version 0.1.0 after the Unicode tokenizer, the
incremental longest-run search and the other fixes listed in the changelog, and every
count in all 24 rows was unchanged. The times are from the earlier run, when the 24 scans
took 2 min 21 s of wall time on an 8-core laptop with a load average between 5 and 12
from other jobs; the rerun took 9 min 52 s at a load average between 30 and 85, so neither
is a benchmark. Among the per-item matches, only the choice between equally long spans
changed: ties now go to the earliest document position.

| eval | corpus | tokenizer | flags | N | gpt3 dirty | palm | llama2 dirty | llama2 not clean | substring | any overlap | scan s | wall s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| humaneval | code_alpaca_20k | word | default | 13 | 7 (4.3 %) | 0 | 0 | 3 (1.8 %) | 0 | 55 | 1.2 | 1.3 |
| humaneval | code_alpaca_20k | word | paper | 13 | 7 (4.3 %) | 1 (0.6 %) | 0 | 3 (1.8 %) | 0 | 55 | 1.4 | 1.5 |
| humaneval | code_alpaca_20k | word | skip4 | 13 | 7 (4.3 %) | 0 | 0 | 3 (1.8 %) | 0 | 55 | 1.4 | 1.5 |
| humaneval | code_alpaca_20k | code | default | 13 | 53 (32.3 %) | 0 | 0 | 18 (11.0 %) | 0 | 139 | 1.5 | 1.6 |
| humaneval | code_alpaca_20k | code | paper | 13 | 48 (29.3 %) | 1 (0.6 %) | 0 | 16 (9.8 %) | 0 | 139 | 1.5 | 1.7 |
| humaneval | code_alpaca_20k | code | skip4 | 13 | 53 (32.3 %) | 0 | 0 | 19 (11.6 %) | 0 | 139 | 1.5 | 1.6 |
| humaneval | self_oss_instruct_50k | word | default | 13 | 15 (9.1 %) | 0 | 0 | 8 (4.9 %) | 0 | 94 | 8.2 | 8.4 |
| humaneval | self_oss_instruct_50k | word | paper | 13 | 14 (8.5 %) | 2 (1.2 %) | 0 | 8 (4.9 %) | 0 | 94 | 7.4 | 7.5 |
| humaneval | self_oss_instruct_50k | word | skip4 | 13 | 15 (9.1 %) | 0 | 0 | 8 (4.9 %) | 0 | 94 | 7.9 | 8.0 |
| humaneval | self_oss_instruct_50k | code | default | 13 | 80 (48.8 %) | 0 | 0 | 52 (31.7 %) | 4 (2.4 %) | 160 | 11.2 | 11.3 |
| humaneval | self_oss_instruct_50k | code | paper | 13 | 76 (46.3 %) | 2 (1.2 %) | 0 | 39 (23.8 %) | 4 (2.4 %) | 160 | 10.8 | 11.0 |
| humaneval | self_oss_instruct_50k | code | skip4 | 13 | 80 (48.8 %) | 0 | 0 | 52 (31.7 %) | 4 (2.4 %) | 160 | 10.2 | 10.4 |
| mbpp_test | code_alpaca_20k | word | default | 13 | 19 (3.8 %) | 0 | 0 | 28 (5.6 %) | 1 (0.2 %) | 163 | 1.3 | 1.4 |
| mbpp_test | code_alpaca_20k | word | paper | 13 | 19 (3.8 %) | 3 (0.6 %) | 0 | 28 (5.6 %) | 1 (0.2 %) | 163 | 1.5 | 1.7 |
| mbpp_test | code_alpaca_20k | word | skip4 | 13 | 19 (3.8 %) | 0 | 0 | 28 (5.6 %) | 1 (0.2 %) | 163 | 1.3 | 1.4 |
| mbpp_test | code_alpaca_20k | code | default | 13 | 88 (17.6 %) | 1 (0.2 %) | 1 (0.2 %) | 83 (16.6 %) | 1 (0.2 %) | 289 | 1.8 | 1.9 |
| mbpp_test | code_alpaca_20k | code | paper | 13 | 85 (17.0 %) | 4 (0.8 %) | 1 (0.2 %) | 79 (15.8 %) | 1 (0.2 %) | 289 | 1.8 | 1.9 |
| mbpp_test | code_alpaca_20k | code | skip4 | 13 | 88 (17.6 %) | 1 (0.2 %) | 1 (0.2 %) | 83 (16.6 %) | 1 (0.2 %) | 289 | 1.5 | 1.6 |
| mbpp_test | self_oss_instruct_50k | word | default | 13 | 12 (2.4 %) | 0 | 0 | 25 (5.0 %) | 0 | 202 | 6.7 | 6.8 |
| mbpp_test | self_oss_instruct_50k | word | paper | 13 | 12 (2.4 %) | 5 (1.0 %) | 0 | 25 (5.0 %) | 0 | 202 | 7.1 | 7.2 |
| mbpp_test | self_oss_instruct_50k | word | skip4 | 13 | 12 (2.4 %) | 0 | 0 | 26 (5.2 %) | 0 | 202 | 8.7 | 8.8 |
| mbpp_test | self_oss_instruct_50k | code | default | 13 | 105 (21.0 %) | 0 | 0 | 108 (21.6 %) | 0 | 330 | 11.3 | 11.4 |
| mbpp_test | self_oss_instruct_50k | code | paper | 13 | 100 (20.0 %) | 0 | 0 | 96 (19.2 %) | 0 | 330 | 12.3 | 12.6 |
| mbpp_test | self_oss_instruct_50k | code | skip4 | 13 | 105 (21.0 %) | 0 | 0 | 110 (22.0 %) | 0 | 330 | 18.0 | 18.2 |
