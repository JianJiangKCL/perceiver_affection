#!/bin/bash



cd ..

SAVE_PATH='results/new_checkpoints'


GPU_ID=0

CONFIG='configs/perceiver_baseline.yaml'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="online"

