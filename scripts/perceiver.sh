#!/bin/bash



cd ..

#SEEDS=(
# 1993
# 1994
# 1995
#
#)
#
#SEED=${SEEDS[0]}
SAVE_PATH='results/second'



GPU_ID=0
DATASET_PATH=''


CONFIG='configs/second_perceiver.yaml'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="online"

