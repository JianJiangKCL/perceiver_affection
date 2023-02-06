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
SAVE_PATH='results/first'



GPU_ID=0
DATASET_PATH=''


CONFIG='configs/perceiver_baseline.yaml'
#['text', 'facebody', 'senti', 'speech']
# create array of modalities in shell script
MODALITIES=('text' 'facebody' 'senti' 'speech')
# iterate over combinations of modalities in shell script
for MODALITY in ${MODALITIES[*]}
do
    echo $MODALITY
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="offline" -t modalities=[$MODALITY]

done


