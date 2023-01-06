# Incremental Bias Mitigation for Fairness  via Multi-Modalities Multi-task Learning




## Project 
The project use pytorch lightning to implement the paper [Incremental Bias Mitigation for Fairness via Multi-Modalities Multi-task Learning]. The perceiver implementation is from https://github.com/fac2003/perceiver-multi-modality-pytorch 
### args
stores the arguments for the project.


one can add custom args in `custom.py`.

### configs
stores `.yaml` config files for experiments.

`sweepxxx.yaml` are for hyperparameter searching.


## Training
the `main.py` is used for all the experiments. It chooses different trainers in the `trainers` folder for different phases.

### First phase
Baseline training. uses `trainers/BaselineTrainer.py` to train a Perceiver model with multi-modality data.

### Second phase
Bias reducing for the first sensitive attribute. uses `trainers/MultiTaskTrainer.py` to train/finetune a Perceiver model with multi-modality data in Multi-Task Learning (MTL) setting. The MTL model has two heads, one for predicting the OCEAN values and the other for matigating bias in the target sensitive group. 


