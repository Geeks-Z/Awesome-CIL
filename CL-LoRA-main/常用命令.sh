cd Code/Research/Awesome-CIL/CL-LoRA-main&&
conda activate cl-peft

nohup ./train_cllora.sh > ../results/CL-LoRA-IN21K-A40-1.out 2>&1 &
