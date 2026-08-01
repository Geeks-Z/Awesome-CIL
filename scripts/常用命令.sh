cd Code/resultsearch/Awesome-CIL&&
conda activate peft
nohup ./scripts/ptm_sh/train_ptm_b16_in1k.sh > ./results/ptm/SimpleCIL-Benchmark-Table1-B16_IN1K.out 2>&1 &
nohup ./scripts/ptm_sh/train_ptm_lora.sh > ./results/PTM-B16-1K-LoRA-A40.out 2>&1 &

nohup ./scripts/train_aper.sh > ./results/APER-LoRA.out 2>&1 &
nohup ./scripts/train_coda_prompt.sh > ./results/CODA_Prompt.out 2>&1 &
nohup ./scripts/train_der.sh > ./results/DER.out 2>&1 &
nohup ./scripts/train_dualprompt.sh > ./results/DualPrompt.out 2>&1 &
nohup ./scripts/train_ease.sh > ./results/EASE-LoRA16.out 2>&1 &
nohup ./scripts/train_finetune.sh > ./results/2nd-Finetune-Supp.out 2>&1 &
nohup ./scripts/train_foster.sh > ./results/FOSTER.out 2>&1 &
nohup ./scripts/train_icarl.sh > ./results/iCaRL.out 2>&1 &
nohup ./scripts/train_l2p.sh > ./results/2nd-L2P-Supp.out 2>&1 &
nohup ./scripts/train_lae.sh > ./results/LAE-LoRA10.out 2>&1 &
nohup ./scripts/train_simplecil.sh > ./results/SimpleCIL.out 2>&1 &
nohup ./scripts/train_inflora.sh > ./results/2nd-InfLoRA-Rank10-Supp.out 2>&1 &
nohup ./scripts/train_sdlora.sh > ./results/2nd-SD-LoRA-Supp.out 2>&1 &
nohup ./scripts/train_bilora.sh > ./results/BiLoRA-Supp.out 2>&1 &
nohup ./scripts/train_cllora.sh > ./results/CL-LoRA-Supp.out 2>&1 &

