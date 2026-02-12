python3 main.py --config=./exps/sdlora_c100.json  >> CIFAR.log 2>&1 &
python3 main.py --config=./exps/sdlora_cub.json  >> CUB.log 2>&1 &
python3 main.py --config=./exps/sdlora_inr.json  >> InR.log 2>&1 &
python3 main.py --config=./exps/sdlora_omni.json  >> Omni.log 2>&1 &
python3 main.py --config=./exps/sdlora_vtab.json  >> VTab.log 2>&1 &
#python3 main.py --config=./scripts/sdlora_ina_B0_Inc20.json  >> InA.log 2>&1 &
