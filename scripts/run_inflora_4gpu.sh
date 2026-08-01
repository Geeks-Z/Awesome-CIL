#!/bin/bash

set -euo pipefail

python main.py --config configs/inflora/inflora_cifar_B0_Inc10.json &
pid1=$!

python main.py --config configs/inflora/inflora_inr_B0_Inc10.json &
pid2=$!

python main.py --config configs/inflora/inflora_inr_B0_Inc20.json &
pid3=$!

python main.py --config configs/inflora/inflora_inr_B0_Inc40.json &
pid4=$!

wait "$pid1"
wait "$pid2"
wait "$pid3"
wait "$pid4"