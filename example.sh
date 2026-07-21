#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-data/KG_data/FB15k-betae}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" main.py \
  --cuda \
  --do_train --do_valid --do_test \
  --data_path "${DATA_PATH}" \
  --negative_sample_size 128 \
  --batch_size 512 \
  --hidden_dim 800 \
  --gamma 24 \
  --learning_rate 0.0001 \
  --max_steps 450001 \
  --valid_steps 15000 \
  --cpu_num 1 \
  --tasks "1p.2p.3p.2i.3i.ip.pi.2u.up"
