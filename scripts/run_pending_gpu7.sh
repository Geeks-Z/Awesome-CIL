#!/usr/bin/env bash
# Run only the still-empty CIL_Results.xlsx cells.  Multiple workers can safely
# share this task list; use a distinct GPU/slot pair for each worker.
# Memory methods (iCaRL/DER) and FOSTER are intentionally excluded.
set -uo pipefail

ROOT="/home/team/zhaohongwei/Code/Research/Awesome-CIL"
DATA_ROOT="/home/team/zhaohongwei/Dataset"
RUN_OUTPUT_ROOT="$ROOT/run_output"
GPU_ID="${1:-}"
WORKER_SLOT="${2:-}"
WORKER_COUNT="${3:-2}"

if ! [[ "$GPU_ID" =~ ^[0-9]+$ && "$WORKER_SLOT" =~ ^[0-9]+$ && "$WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] \
  || (( WORKER_SLOT >= WORKER_COUNT )); then
  echo "Usage: $0 <gpu-id> <worker-slot> [worker-count]" >&2
  echo "Example: $0 6 0 2   # first of two workers, using GPU 6" >&2
  exit 2
fi

cd "$ROOT"
source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate peft
export CIL_DATA_ROOT="$DATA_ROOT"

run_one() {
  local config="$1"
  local seed="$2"
  local destination="$3"
  local lock_dir="$RUN_OUTPUT_ROOT/locks"
  local lock_path
  local owner_pid
  local status

  mkdir -p "$lock_dir"
  lock_path="$lock_dir/${destination}_$(basename "$config" .json)_${seed}.lock"
  if ! mkdir "$lock_path" 2> /dev/null; then
    owner_pid=""
    if [ -f "$lock_path/pid" ]; then
      read -r owner_pid < "$lock_path/pid" || true
    fi
    if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2> /dev/null; then
      echo "SKIP active experiment lock: $lock_path (pid $owner_pid)"
      return 0
    fi
    echo "RECLAIM stale experiment lock: $lock_path"
    rm -rf "$lock_path"
    if ! mkdir "$lock_path" 2> /dev/null; then
      echo "SKIP contested experiment lock: $lock_path"
      return 0
    fi
  fi
  printf '%s\n' "$$" > "$lock_path/pid"

  python - "$config" "$seed" "$destination" "$DATA_ROOT" "$GPU_ID" <<'PY'
import ast
import json
import os
import re
import sys

from trainer import train
from utils.logging_utils import canonical_log_dir

config_path, seed, destination, data_root, gpu_id = sys.argv[1:]
seed = int(seed)
with open(config_path) as handle:
    args = json.load(handle)

dataset_roots = {
    "cifar224": data_root,
    "cub": os.path.join(data_root, "cub"),
    "imagenetr": os.path.join(data_root, "imagenet-r"),
    "imagenet_r": os.path.join(data_root, "imagenet-r"),
    "omnibenchmark": os.path.join(data_root, "omnibenchmark"),
    "vtab": os.path.join(data_root, "vtab"),
}
args["data_path"] = dataset_roots.get(args["dataset"], data_root)
args["device"] = [gpu_id]
args["seed"] = [seed]

backbone = args["backbone_type"].removeprefix("pretrained_")
init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
filename = "{}_{}_{}_{}_{}.log".format(
    args["dataset"], backbone, init_cls, args["increment"], seed
)
source = os.path.join("logs", canonical_log_dir(args["model_name"]), filename)


def complete(path):
    if not os.path.isfile(path):
        return False
    text = open(path, errors="ignore").read()
    curves = re.findall(r"CNN top1 curve: (\[[^\r\n]+\])", text)
    if not curves:
        return False
    try:
        curve = ast.literal_eval(curves[-1])
    except (SyntaxError, ValueError):
        return False
    class_counts = {
        "cifar224": 100,
        "cub": 200,
        "imagenetr": 200,
        "imagenet_r": 200,
        "omnibenchmark": 300,
        "vtab": 50,
    }
    class_count = class_counts.get(args["dataset"])
    if class_count is None:
        return False
    initial_classes = args["init_cls"] or args["increment"]
    expected_tasks = 1 + (class_count - initial_classes + args["increment"] - 1) // args["increment"]
    return len(curve) == expected_tasks and "Average Accuracy (CNN)" in text


if complete(source):
    print("SKIP complete log: {}".format(source), flush=True)
    raise SystemExit(0)

print(
    "START config={} seed={} device={} data_path={}".format(
        config_path, seed, args["device"][0], args["data_path"]
    ),
    flush=True,
)
train(args)

if complete(source):
    print("COMPLETE: {}".format(source), flush=True)
else:
    print("INCOMPLETE: {}".format(source), flush=True)
    raise SystemExit(1)
PY
  status=$?
  rm -rf "$lock_path"
  return "$status"
}

failures=0
task_index=0
selected=0
submit() {
  local current_index="$task_index"
  task_index=$((task_index + 1))
  if (( current_index % WORKER_COUNT != WORKER_SLOT )); then
    return 0
  fi
  selected=$((selected + 1))
  if ! run_one "$@"; then
    echo "FAILED: config=$1 seed=$2 destination=$3" >&2
    failures=$((failures + 1))
  fi
}

# LAE: ImageNet-R T=5 (200 classes / Inc40), seeds 42/1993/1994.
for seed in 42 1993 1994; do
  submit configs/lae/lae_inr_B0_Inc40.json "$seed" LAE
done

# InfLoRA: CUB200 T=10 (200 classes / Inc20), seed 1993.
submit configs/inflora/inflora_cub_B0_Inc20.json 1993 InfLoRA

# BiLoRA: OmniBenchmark T=10 (42/1994) and VTAB T=5 (42/1993/1994).
for seed in 42 1994; do
  submit configs/bilora/bilora_omni_B0_Inc30.json "$seed" BiLoRA
done
for seed in 42 1993 1994; do
  submit configs/bilora/bilora_vtab_B0_Inc10.json "$seed" BiLoRA
done

# SD-LoRA: CIFAR100, seed 42.
submit configs/sdlora/sdlora_cifar_B0_Inc10.json 42 SD-LoRA-IN21K

# CL-LoRA: VTAB T=5 (50 classes / Inc10), seeds 42/1994.
for seed in 42 1994; do
  submit configs/cllora/cllora_vtab_B0_Inc10.json "$seed" CL-LoRA
done

if [ "$failures" -ne 0 ]; then
  echo "GPU $GPU_ID worker $WORKER_SLOT: $failures of $selected selected experiment(s) failed; inspect this launcher log." >&2
  exit 1
fi

echo "GPU $GPU_ID worker $WORKER_SLOT: processed $selected assigned experiment(s) without launcher failures."
