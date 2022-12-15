#!/bin/bash



cd ..

SEEDS=(
 1993
 1994
 1995

)

SEED=${SEEDS[0]}
SAVE_PATH='results'

GPU_ID=0
DATASET_PATH=''


CONFIG='configs/perceiver.yaml'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG} --dataset_path ${DATASET_PATH} --results_dir $SAVE_PATH  --seed $SEED   --wandb_mode offline

