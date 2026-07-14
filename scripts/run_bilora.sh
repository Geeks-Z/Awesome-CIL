#!/bin/bash
set -euo pipefail

mkdir -p logs/bilora

python main.py --config ./configs/bilora/bilora_cub_B0_Inc20.json > ./logs/bilora/bilora_cub_B0_Inc20.out 2>&1 &
PID_CUB=$!

python main.py --config ./configs/bilora/bilora_omni_B0_Inc30.json > ./logs/bilora/bilora_omni_B0_Inc30.out 2>&1 &
PID_OMNI=$!

python main.py --config ./configs/bilora/bilora_vtab_B0_Inc10.json > ./logs/bilora/bilora_vtab_B0_Inc10.out 2>&1 &
PID_VTAB=$!

wait $PID_CUB
wait $PID_OMNI
wait $PID_VTAB

echo "BiLoRA finished."
