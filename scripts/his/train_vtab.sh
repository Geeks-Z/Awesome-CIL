#!/bin/bash
python main.py --config=./scripts/finetune/finetune_vtab_B0_Inc5.json
python main.py --config=./scripts/finetune/finetune_vtab_B0_Inc10.json
python main.py --config=./scripts/l2p/l2p_vtab_B0_Inc5.json
python main.py --config=./scripts/l2p/l2p_vtab_B0_Inc10.json
python main.py --config=./scripts/dualprompt/dualprompt_vtab_B0_Inc5.json
python main.py --config=./scripts/dualprompt/dualprompt_vtab_B0_Inc10.json
python main.py --config=./scripts/coda_prompt/coda_prompt_vtab_B0_Inc5.json
python main.py --config=./scripts/coda_prompt/coda_prompt_vtab_B0_Inc10.json
python main.py --config=./scripts/simplecil/simplecil_vtab_B0_Inc5.json
python main.py --config=./scripts/simplecil/simplecil_vtab_B0_Inc10.json
python main.py --config=./scripts/ease/ease_vtab_B0_Inc5.json
python main.py --config=./scripts/ease/ease_vtab_B0_Inc10.json




