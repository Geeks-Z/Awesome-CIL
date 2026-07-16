#!/usr/bin/env bash
set -u

cd /public/home/hanlida/Dr.1/Code/Research/Awesome-CIL
source /public/home/hanlida/ld/etc/profile.d/conda.sh
conda activate il

run_one() {
  local config="$1"
  local seed="$2"
  local destination="$3"
  python - "$config" "$seed" "$destination" <<'PY'
import json
import os
import shutil
import sys

from trainer import train

config_path, seed, destination = sys.argv[1], int(sys.argv[2]), sys.argv[3]
args = json.load(open(config_path))
backbone = args['backbone_type'].removeprefix('pretrained_')
init_cls = 0 if args['init_cls'] == args['increment'] else args['init_cls']
filename = f"{args['dataset']}_{backbone}_{init_cls}_{args['increment']}_{seed}.log"
source = os.path.join('logs', args['model_name'], filename)
target_dir = os.path.join('logs', '第二轮A800', destination)
target = os.path.join(target_dir, filename)

def complete(path):
    return os.path.exists(path) and 'Average Accuracy' in open(path, errors='ignore').read()

if complete(target):
    print(f'SKIP complete target: {target}', flush=True)
    raise SystemExit(0)
if complete(source):
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(source, target)
    print(f'MOVED complete log: {target}', flush=True)
    raise SystemExit(0)

args['seed'] = [seed]
args['device'] = ['3']
train(args)

if complete(source):
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(source, target)
    print(f'COMPLETE and moved: {target}', flush=True)
else:
    print(f'INCOMPLETE: {source}', flush=True)
    raise SystemExit(1)
PY
}

# Required order: BiLoRA > CL-LoRA > LAE > iCaRL > DER > FOSTER.
for seed in 42 1993 1994; do run_one configs/bilora/bilora_vtab_B0_Inc5.json "$seed" BiLoRA; done
for seed in 42 1993 1994; do run_one configs/cllora/cllora_vtab_B0_Inc5.json "$seed" CL-LoRA; done
for seed in 42 1993 1994; do run_one configs/lae/lae_inr_B0_Inc5.json "$seed" LAE; done
for cfg in configs/icarl/icarl_inr_B0_Inc5.json configs/icarl/icarl_vtab_B0_Inc5.json; do
  for seed in 42 1993 1994; do run_one "$cfg" "$seed" iCaRL; done
done
for cfg in configs/der/*.json; do
  for seed in 42 1993 1994; do run_one "$cfg" "$seed" DER; done
done
for cfg in configs/foster/foster_cifar_B0_Inc10.json configs/foster/foster_cub_B0_Inc10.json configs/foster/foster_omn_B0_Inc10.json; do
  for seed in 42 1993 1994; do run_one "$cfg" "$seed" FOSTER; done
done
