#!/usr/bin/env bash
# Scan HumanEval and the MBPP test split against two public code SFT sets, with both
# tokenisers, in three settings: the default flags ("default"), the GPT-3 and PaLM
# variants --gpt3-n auto --palm-per-field --max-doc-freq 10 ("paper"), and the
# approximate Llama 2 skipgram budget --llama-skip-budget 4 ("skip4").
#
# Usage: examples/run_code_audit.sh DATA_DIR OUT_DIR [WORKERS]
# DATA_DIR holds raw/ (from fetch_code_data.sh) and jsonl/ (from prepare_code_data.py).
set -euo pipefail
data=$1
out=$2
workers=${3:-4}
mkdir -p "$out"
: > "$out/times.tsv"
for ev in humaneval mbpp_test; do
  for corpus in code_alpaca_20k self_oss_instruct_50k; do
    if [ "$corpus" = code_alpaca_20k ]; then
      cpath=$data/jsonl/code_alpaca_20k.jsonl
      cfields=instruction,input,output
    else
      cpath=$data/raw/self_oss_instruct_50k.parquet
      cfields=instruction,response
    fi
    for tok in word code; do
      for variant in default paper skip4; do
        extra=()
        if [ "$variant" = paper ]; then
          extra=(--gpt3-n auto --palm-per-field --max-doc-freq 10)
        elif [ "$variant" = skip4 ]; then
          extra=(--llama-skip-budget 4)
        fi
        name=$ev-$corpus-$tok-$variant
        start=$(python3 -c 'import time; print(time.time())')
        contamscan scan "$data/jsonl/$ev.jsonl" "$cpath" --fields prompt,solution \
          --corpus-fields "$cfields" --tokenizer "$tok" --workers "$workers" \
          "${extra[@]+"${extra[@]}"}" \
          --json "$out/$name.json" --markdown "$out/$name.md" > "$out/$name.txt"
        end=$(python3 -c 'import time; print(time.time())')
        wall=$(python3 -c "print(f'{$end - $start:.1f}')")
        printf '%s\t%s\n' "$name" "$wall" >> "$out/times.tsv"
        echo "$name: ${wall}s"
      done
    done
  done
done
