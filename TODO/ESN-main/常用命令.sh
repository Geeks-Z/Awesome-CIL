cd Code/Research/Awesome-CIL/ESN-main&&
conda activate peft

nohup ./train.sh > ./res/B0-prompt-cub-ina-omn-vtab-id-50epoch.out 2>&1 &
