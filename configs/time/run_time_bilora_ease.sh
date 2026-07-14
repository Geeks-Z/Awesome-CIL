#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/time

python main.py --config configs/time/bilora_cifar_B0_Inc10.json > logs/time/bilora_cifar_B0_Inc10_gpu4.log 2>&1 &
python main.py --config configs/time/bilora_inr_B0_Inc40.json > logs/time/bilora_inr_B0_Inc40_gpu5.log 2>&1 &
python main.py --config configs/time/ease_cifar_B0_Inc10.json > logs/time/ease_cifar_B0_Inc10_gpu6.log 2>&1 &
python main.py --config configs/time/ease_inr_B0_Inc40.json > logs/time/ease_inr_B0_Inc40_gpu7.log 2>&1 &

wait
