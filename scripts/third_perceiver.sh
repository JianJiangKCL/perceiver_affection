#!/bin/bash



cd ..

SAVE_PATH='results/third'


GPU_ID=2
DATASET_PATH=''


CONFIG='configs/third_perceiver.yaml'

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="online"

