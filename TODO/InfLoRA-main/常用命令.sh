cd Code/Research/Awesome-CIL/TODO/InfLoRA-main&&
conda activate peft

nohup ./train.sh > ./res/inflora-B0-big.out 2>&1 &

nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &