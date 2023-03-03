#!/bin/bash



cd ..

SAVE_PATH='fiv2_results/second'


GPU_ID=0
DATASET_PATH=''
SEEDS=( 1995  6 1996)

CONFIG='configs/fiv2/second_perceiver.yaml'
modes=("audio")
FINETUNE=/jmain02/home/J2AD019/exk04/jxj51-exk04/proj/perceiver_affection/fiv2_results/baseline/gender/xxx_lr0.004_e160_seedsss_optlamb_bs128_beta0.5_alpha_0.1_gamma_1/last.ckpt

for mode in ${modes[*]}
  do
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -c ${CONFIG}  --results_dir $SAVE_PATH -t wandb_mode="offline" --finetune ${FINETUNE} -t modalities=$mode -t target_sensitive_group="ethnicity"  -t seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}
done
