cd Code/Research/Awesome-CIL/InfLoRA-main&&
conda activate peft

nohup ./train_inflora.sh > ./res/inflora-1-ina.out 2>&1 &

nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &