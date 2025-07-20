#!/bin/bash
python main.py --config=./scripts/prompt/l2p_cifar_B0_Inc10_B16_IN1K.json
python main.py --config=./scripts/prompt/l2p_cub_B0_Inc20_B16_IN1K.json
python main.py --config=./scripts/prompt/l2p_inr_B0_Inc40_B16_IN1K.json
python main.py --config=./scripts/prompt/l2p_omn_B0_Inc30_B16_IN1K.json
python main.py --config=./scripts/prompt/l2p_vtab_B0_Inc10_B16_IN1K.json
python main.py --config=./scripts/prompt/dualprompt_cifar_B0_Inc10_B16_IN1K.json
python main.py --config=./scripts/prompt/dualprompt_cub_B0_Inc20_B16_IN1K.json
python main.py --config=./scripts/prompt/dualprompt_inr_B0_Inc40_B16_IN1K.json
python main.py --config=./scripts/prompt/dualprompt_omn_B0_Inc30_B16_IN1K.json
python main.py --config=./scripts/prompt/dualprompt_vtab_B0_Inc10_B16_IN1K.json



