#!/bin/bash



cd ..

SAVE_PATH='fiv2_results/baseline'



GPU_ID=0

SEEDS=(1995  6 1996 1997)

CONFIG='configs/fiv2/adv_personality.yaml'
modes=("audio")
#FINETUNE=
for mode in ${modes[*]}
  do
#    echo $mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="offline" -t modalities=$mode -t seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
  done


