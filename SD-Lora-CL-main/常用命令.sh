cd Code/Research/Awesome-CIL/SD-Lora-CL-main&&
conda activate peft

nohup ./train_sdlora.sh > ../results/SD-LoRA-IN21K-rank4.out 2>&1 &