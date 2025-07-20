cd Code/Research/Awesome-CIL/InfLoRA-main&&
conda activate peft

nohup ./train.sh > ./res/.out 2>&1 &
nohup ./train_inflora.sh > ../res/inflora-rank10-ViTB16-IN1K.out 2>&1 &

nohup ./loop_B0.sh > ./res/loop_B0.out 2>&1 &
nohup ./loop_B50.sh > ./res/loop_B50.out 2>&1 &