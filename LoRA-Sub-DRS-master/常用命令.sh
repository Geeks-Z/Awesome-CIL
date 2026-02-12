cd Code/Research/Awesome-CIL/LoRA-Sub-DRS-master&&
conda activate cl-peft

nohup ./train_lorasub.sh > ../results/LoRA-Sub-DRS-IN21K-A40-1.out 2>&1 &
