cd Code/Research/Awesome-CIL/SD-Lora-CL-main&&
conda activate peft

nohup ./train_sdlora.sh > ../results/SD-LoRA-IN21K-A40-1.out 2>&1 &

nohup ./loop_B0.sh > ./results/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./results/loop_B50.out 2>&1 &