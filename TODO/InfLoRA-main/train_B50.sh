#!/bin/bash
python main.py --device '1' --config configs/inflora_cifar_B50_Inc5.json
python main.py --device '1' --config configs/inflora_cub_B100_Inc5.json
python main.py --device '1' --config configs/inflora_inr_B100_Inc5.json
python main.py --device '1' --config configs/inflora_omn_B150_Inc5.json
