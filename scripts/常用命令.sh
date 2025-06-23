cd Code/Research/Awesome-CIL&&
conda activate peft
# 4090
cd lrh/Code/Research/CIL/LAMDA-PILOT/ &&
conda activate cil
# 超算
cd Code/Research/CIL/LAMDA-PILOT/ &&
sbatch train.slurm

#nohup ./scripts/train_aper.sh > ./res/aper-adapter.out 2>&1 &
#nohup ./scripts/train_coda_prompt.sh > ./res/coda_prompt-vtab.out 2>&1 &
#nohup ./scripts/train_der.sh > ./res/der.out 2>&1 &
#nohup ./scripts/train_dualprompt.sh > ./res/dualprompt2-ina.out 2>&1 &
#nohup ./scripts/train_ease.sh > ./res/ease.out 2>&1 &
#nohup ./scripts/train_finetune.sh > ./res/finetune-2-cub.out 2>&1 &
#nohup ./scripts/train_foster.sh > ./res/foster.out 2>&1 &
#nohup ./scripts/train_icarl.sh > ./res/icarl-1-cub.out 2>&1 &
#nohup ./scripts/train_l2p.sh > ./res/l2p.out 2>&1 &
#nohup ./scripts/train_lae.sh > ./res/lae.out 2>&1 &
#nohup ./scripts/train_mos.sh > ./res/mos-4.out 2>&1 &
#nohup ./scripts/train_simplecil.sh > ./res/simplecil.out 2>&1 &

-----------------------------------------------------------------------------------
nohup ./scripts/train_simplecil.sh > ./res/v21.out 2>&1 &

nohup ./scripts/train_lae.sh > ./res/lae-B0-big.out 2>&1 &


nohup ./scripts/train_coil.sh > ./res/B0-coil-omn.out 2>&1 &

nohup ./aper_loop.sh > ./res/aper_loop_B0.out 2>&1 &

nohup ./lae_loop.sh > ./res/lae_loop_B0.out 2>&1 &

---------------------------------------------------------------------------------------------------

nohup ./scripts/train_l2p.sh > ./res/l2p-time.out 2>&1 &
nohup ./scripts/train_lae.sh > ./res/lae-time.out 2>&1 &
nohup ./scripts/train_aper.sh > ./res/aper-adapter-time.out 2>&1 &
nohup ./scripts/train_coda_prompt.sh > ./res/coda_prompt-time.out 2>&1 &
nohup ./scripts/train_dualprompt.sh > ./res/dualprompt-time.out 2>&1 &
nohup ./scripts/train_der.sh > ./res/der1-supp.out 2>&1 &










----------------------------------------------------------------------------------------------------------
#cifar
nohup ./train_cifar.sh > ./log/cifar-stdout.log 2> ./log/cifar-stderr.log &
nohup ./train_cifar.sh > ./res/cifar.out 2>&1 &
#cub
nohup ./train_cub.sh > ./log/cub-stdout.log 2> ./log/cub-stderr.log &
nohup ./train_cub.sh > ./res/cub.out 2>&1 &
#inr
nohup ./train_inr.sh > ./log/inr-stdout.log 2> ./log/inr-stderr.log &
nohup ./train_inr.sh > ./res/inr.out 2>&1 &
#ina
nohup ./train_ina.sh > ./log/ina-stdout.log 2> ./log/ina-stderr.log &
nohup ./train_ina.sh > ./res/ina.out 2>&1 &
#omn
nohup ./train_omn.sh > ./log/omn-stdout.log 2> ./log/omn-stderr.log &
nohup ./train_omn.sh > ./res/omn.out 2>&1 &
#vtab
nohup ./train_vtab.sh > ./log/vtab-stdout.log 2> ./log/vtab-stderr.log &
nohup ./train_vtab.sh > ./res/vtab.out 2>&1 &
----------------------------------------------------------------------------------------------------------