cd Code/Research/Awesome-CIL/InfLoRA-main&&
conda activate peft

nohup ./train_inflora.sh > ../results/InfLoRA-rank4-ViTB16-IN21K.out 2>&1 &
