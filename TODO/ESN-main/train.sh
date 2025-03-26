#!/bin/bash
#python main.py --dataset cifar100_vit --max_epochs 30 --init_cls 20 --inc_cls 20 --shuffle
#python main.py --dataset imagenetr --max_epochs 30 --init_cls 40 --inc_cls 40 --shuffle
python main.py --dataset cub --max_epochs 50 --init_cls 20 --inc_cls 20 --shuffle
python main.py --dataset imageneta --max_epochs 50 --init_cls 10 --inc_cls 10 --shuffle
python main.py --dataset omnibenchmark --max_epochs 50 --init_cls 10 --inc_cls 10 --shuffle
python main.py --dataset vtab --max_epochs 50 --init_cls 5 --inc_cls 5 --shuffle