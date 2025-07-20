cd Code/Research/Awesome-CIL/SD-Lora-CL-main&&
conda activate peft

nohup ./train_sdlora.sh > ../res/sdlora-A40-ViT_B16IN21K.out 2>&1 &

nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &