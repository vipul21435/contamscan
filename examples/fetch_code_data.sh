#!/usr/bin/env bash
# Download the public files used in the README results (about 8.3 MB in total).
# HumanEval: MIT; MBPP: CC BY 4.0; CodeAlpaca-20k: CC BY 4.0 (licences as stated on
# each dataset card on Hugging Face).
set -euo pipefail
dest=${1:-data/raw}
mkdir -p "$dest"
hf=https://huggingface.co/datasets
curl -sSL -o "$dest/humaneval.parquet" \
  "$hf/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet"
curl -sSL -o "$dest/mbpp_full_test.parquet" \
  "$hf/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet"
curl -sSL -o "$dest/code_alpaca_20k.json" \
  "$hf/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json"
ls -l "$dest"
