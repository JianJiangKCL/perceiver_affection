#!/bin/bash



cd ..

SAVE_PATH='results/third'



GPU_ID=0

SEEDS=(6 1995 1996 )

CONFIG='configs/third_perceiver.yaml'
modes=("text"  "facebody"  "audio"  "senti_speech_time"  "text_facebody"  "text_audio"  "text_senti_speech_time"  "facebody_audio"  "facebody_senti_speech_time"  "audio_senti_speech_time"  "text_facebody_audio"  "text_facebody_senti_speech_time"  "text_audio_senti_speech_time"  "facebody_audio_senti_speech_time"  "text_facebody_audio_senti_speech_time")
GROUPS=('age' 'gender')
FINETUNE=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/perceiver_affection/results/second/ttt/xxx_lr0.004_e5_seedsss_optlamb_bs128_beta0.5_alpha_0.1_gamma_5/last.ckpt
for mode in ${modes[*]}
  do
#    echo $mode
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="offline" --finetune ${FINETUNE} -t modalities=$mode -t seed=${SEEDS[$SLURM_ARRAY_TASK_ID]} -t target_sensitive_group='age' #-t target_sensitive_group=${GROUPS[$SLURM_ARRAY_TASK_ID]} #
  done


