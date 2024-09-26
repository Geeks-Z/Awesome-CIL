cd Code/Research/CIL/LAMDA-PILOT/&&
conda activate cil

#cifar
nohup ./train.sh > ./log/3rd-cifar-stdout.log 2> ./log/3rd-cifar-stderr.log &
#ina
nohup ./train_ina.sh > ./log/3rd-ina-stdout.log 2> ./log/3rd-ina-stderr.log &
#inr
nohup ./train_inr.sh > ./log/inr-stdout.log 2> ./log/inr-stderr.log &
#cub
nohup ./train_cub.sh > ./log/err-cub-stdout.log 2> ./log/err-cub-stderr.log &


----------------------------------------------------------------------------------------------------------
nohup ./train.sh > ./res/supp-cifar.out 2>&1 &

# cub
nohup ./train_cub.sh > ./res/supp-cub.out 2>&1 &
# ina
nohup ./train_ina.sh > ./res/4th-ina.out 2>&1 &
# inr
nohup ./train_inr.sh > ./res/3nd_inr.out 2>&1 &
# omn
nohup ./train_omn.sh > ./res/2nd-omn.out 2>&1 &
# vtab
nohup ./train_vtab.sh > ./res/3nd-vtab.out 2>&1 &



# ease
nohup ./train.sh > ./res/ease-all-datasets.out 2>&1 &
# 4090
cd lrh/Code/Research/CIL/LAMDA-PILOT/ &&
conda activate cil

# 超算
cd Code/Research/CIL/LAMDA-PILOT/ &&
sbatch train.slurm




# der cifar
nohup ./train_der_cifar.sh > ./res/4090-der-cifar.out 2>&1 &

# foster cifar
nohup ./train_foster_cifar.sh > ./res/4090-foster-cifar.out 2>&1 &

# memory对比

nohup ./train_vtab_B0_Inc5.sh >./res/memory-baseline-vtab-B0-Inc5.out 2>&1 &

nohup ./train_cifar_B0_Inc10.sh >./res/memory-baseline-cifar-B0-Inc10.out 2>&1 &

nohup ./train_cub_B0_Inc5.sh >./res/memory-baseline-cub-B0-Inc5.out 2>&1 &

nohup ./train_omn_B0_Inc10.sh >./res/memory-baseline-omn-B0-Inc10.out 2>&1 &

-------
nohup ./train_omn_B0_Inc30.sh > ./res/2nd-baseline-omn-B0-Inc30-ease.out 2>&1 &

nohup ./train_omn_B150_Inc5.sh > ./res/2nd-baseline-omn-B150-Inc5.out 2>&1 &

nohup ./train_cub_B100_Inc5.sh > ./res/2nd-baseline-cub-B100-Inc5.out 2>&1 &

nohup ./train_omn_B0_Inc30.sh > ./res/3nd-baseline-omn-B0-Inc30.out 2>&1 &

nohup ./train_cifar_B0_Inc5.sh > ./res/3nd-baseline-cifar-B0-Inc5.out 2>&1 &

nohup ./train_cub_B0_Inc10.sh > ./res/3nd-baseline-cub-B0-Inc10.out 2>&1 &

nohup ./train_inr_B0_Inc5.sh > ./res/3nd-baseline-inr-B0-Inc5.out 2>&1 &

nohup ./train_ina_B0_Inc20.sh > ./res/3nd-baseline-ina-B0-Inc20.out 2>&1 &

nohup ./train_vtab_B0_Inc10.sh > ./res/3nd-baseline-vtab-B0-Inc10.out 2>&1 &