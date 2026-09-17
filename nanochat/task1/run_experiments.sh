#!/usr/bin/env bash
set -e

# Set base directory
export NANOCHAT_BASE_DIR="/local/s4859049/nanochat_cache"
export PYTHONPATH=".:$PYTHONPATH"

echo "=========================================================="
echo "Experiment 1: Training Tokenizer 8,192 on 500 MB Sample"
echo "=========================================================="
python3 assignments/task1/train_tokenizer.py --vocab-size 8192 --max-chars 500000000 --output-dir tokenizer_8k

echo ""
echo "=========================================================="
echo "Experiment 2: Training Tokenizer 32,768 on 500 MB Sample"
echo "=========================================================="
python3 assignments/task1/train_tokenizer.py --vocab-size 32768 --max-chars 500000000 --output-dir tokenizer_32k

echo ""
echo "=== Training Finished Successfully for Both Tokenizers ==="