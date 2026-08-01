#!/bin/bash
#SBATCH -J omn #作业名称
#SBATCH -p gpu-quota # 提交到指定分区
#SBATCH -N 1 # 使用一个节点
#SBATCH -n 8 # 使用进程（cpu 核）
#SBATCH --gres=gpu:1
#SBATCH -o ./res/3rd-omn-stdout.log
#SBATCH -e ./res/3rd-omn-stderr.log

module load anacoda3
source activate cil

python main.py --config=./scripts/finetune/finetune_omn_B0_Inc5.json
python main.py --config=./scripts/finetune/finetune_omn_B0_Inc10.json
python main.py --config=./scripts/finetune/finetune_omn_B0_Inc20.json
python main.py --config=./scripts/finetune/finetune_omn_B0_Inc30.json
python main.py --config=./scripts/finetune/finetune_omn_B150_Inc5.json
python main.py --config=./scripts/finetune/finetune_omn_B150_Inc10.json
python main.py --config=./scripts/icarl/icarl_omn_B0_Inc5.json
python main.py --config=./scripts/icarl/icarl_omn_B0_Inc10.json
python main.py --config=./scripts/icarl/icarl_omn_B0_Inc20.json
python main.py --config=./scripts/icarl/icarl_omn_B0_Inc30.json
python main.py --config=./scripts/icarl/icarl_omn_B150_Inc5.json
python main.py --config=./scripts/icarl/icarl_omn_B150_Inc10.json
python main.py --config=./scripts/der/der_omn_B0_Inc5.json
python main.py --config=./scripts/der/der_omn_B0_Inc10.json
python main.py --config=./scripts/der/der_omn_B0_Inc20.json
python main.py --config=./scripts/der/der_omn_B0_Inc30.json
python main.py --config=./scripts/der/der_omn_B150_Inc5.json
python main.py --config=./scripts/der/der_omn_B150_Inc10.json
python main.py --config=./scripts/foster/foster_omn_B0_Inc5.json
python main.py --config=./scripts/foster/foster_omn_B0_Inc10.json
python main.py --config=./scripts/foster/foster_omn_B0_Inc20.json
python main.py --config=./scripts/foster/foster_omn_B0_Inc30.json
python main.py --config=./scripts/foster/foster_omn_B150_Inc5.json
python main.py --config=./scripts/foster/foster_omn_B150_Inc10.json
python main.py --config=./scripts/l2p/l2p_omn_B0_Inc5.json
python main.py --config=./scripts/l2p/l2p_omn_B0_Inc10.json
python main.py --config=./scripts/l2p/l2p_omn_B0_Inc20.json
python main.py --config=./scripts/l2p/l2p_omn_B0_Inc30.json
python main.py --config=./scripts/l2p/l2p_omn_B150_Inc5.json
python main.py --config=./scripts/l2p/l2p_omn_B150_Inc10.json
python main.py --config=./scripts/dualprompt/dualprompt_omn_B0_Inc5.json
python main.py --config=./scripts/dualprompt/dualprompt_omn_B0_Inc10.json
python main.py --config=./scripts/dualprompt/dualprompt_omn_B0_Inc20.json
python main.py --config=./scripts/dualprompt/dualprompt_omn_B0_Inc30.json
python main.py --config=./scripts/dualprompt/dualprompt_omn_B150_Inc5.json
python main.py --config=./scripts/dualprompt/dualprompt_omn_B150_Inc10.json
python main.py --config=./scripts/coda_prompt/coda_prompt_omn_B0_Inc5.json
python main.py --config=./scripts/coda_prompt/coda_prompt_omn_B0_Inc10.json
python main.py --config=./scripts/coda_prompt/coda_prompt_omn_B0_Inc20.json
python main.py --config=./scripts/coda_prompt/coda_prompt_omn_B0_Inc30.json
python main.py --config=./scripts/coda_prompt/coda_prompt_omn_B150_Inc5.json
python main.py --config=./scripts/coda_prompt/coda_prompt_omn_B150_Inc10.json
python main.py --config=./scripts/simplecil/simplecil_omn_B0_Inc5.json
python main.py --config=./scripts/simplecil/simplecil_omn_B0_Inc10.json
python main.py --config=./scripts/simplecil/simplecil_omn_B0_Inc20.json
python main.py --config=./scripts/simplecil/simplecil_omn_B0_Inc30.json
python main.py --config=./scripts/simplecil/simplecil_omn_B150_Inc5.json
python main.py --config=./scripts/simplecil/simplecil_omn_B150_Inc10.json
python main.py --config=./scripts/ease/ease_omn_B0_Inc5.json
python main.py --config=./scripts/ease/ease_omn_B0_Inc10.json
python main.py --config=./scripts/ease/ease_omn_B0_Inc20.json
python main.py --config=./scripts/ease/ease_omn_B0_Inc30.json
python main.py --config=./scripts/ease/ease_omn_B150_Inc5.json
python main.py --config=./scripts/ease/ease_omn_B150_Inc10.json
