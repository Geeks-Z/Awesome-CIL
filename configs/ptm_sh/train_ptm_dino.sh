#!/bin/bash
python main.py --config ./configs/simplecil/dino/simplecil_cifar_B0_Inc10_DINO16.json
python main.py --config ./configs/simplecil/dino/simplecil_cub_B0_Inc20_DINO16.json
python main.py --config ./configs/simplecil/dino/simplecil_inr_B0_Inc40_DINO16.json
python main.py --config ./configs/simplecil/dino/simplecil_omni_B0_Inc30_DINO16.json
python main.py --config ./configs/simplecil/dino/simplecil_vtab_B0_Inc10_DINO16.json