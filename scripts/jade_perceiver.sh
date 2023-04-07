#!/bin/bash



cd ..

SAVE_PATH='results/baseline_age26'



GPU_ID=0

SEEDS=( 1995 0 6 1996 1997)

CONFIG='configs/adv_personality.yaml'
modes=("text"  "facebody"  "audio"  "senti_speech_time"  "text_facebody"  "text_audio"  "text_senti_speech_time"  "facebody_audio"  "facebody_senti_speech_time"  "audio_senti_speech_time"  "text_facebody_audio"  "text_facebody_senti_speech_time"  "text_audio_senti_speech_time"  "facebody_audio_senti_speech_time"  "text_facebody_audio_senti_speech_time")

for mode in ${modes[*]}
  do
#    echo $mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="offline" -t modalities=$mode -t seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
  done


