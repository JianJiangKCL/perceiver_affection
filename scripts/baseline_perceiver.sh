#!/bin/bash



cd ..

SAVE_PATH='results/baseline'


GPU_ID=1
DATASET_PATH=''


CONFIG='configs/perceiver_baseline.yaml'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="online"

