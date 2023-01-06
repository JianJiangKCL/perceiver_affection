import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import KnowledgeDistillationLoss
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap, FairnessDistributionLoss, entropy_loss_func
import wandb


class MultiTaskTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities):
        super(MultiTaskTrainer, self).__init__(args, backbone, modalities)
        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()
        self.metrics = {'train': self.train_metric, 'val': self.val_metric, 'test': self.test_metric}
        self.mse_loss = nn.MSELoss()
        self.cls_imbalance_loss = nn.CrossEntropyLoss()
        self.fairness_loss1 = FairnessDistributionLoss()
        self.fairness_loss = None

    def shared_step(self, batch, mode):
        x, label_ocean, label_sen = batch
        label_sen = label_sen.long()
        modalities_x = {modality: x[modality] for modality in self.modalities}

        fv = self.backbone.extract_features(modalities_x)
        pred_ocean = self.backbone.to_logits(fv)
        pred_sen = self.backbone.cosine_fc(fv)

        loss_ocean = self.mse_loss(pred_ocean, label_ocean)
        self.metrics[mode].update(pred_ocean, label_ocean)
        metric = self.metrics[mode].compute()

        log_data = {
            f'{mode}_loss_ocean': loss_ocean,
        }

        loss = loss_ocean
        if self.current_epoch == 9:
            k = 1
        if self.args.use_distribution_loss:
            loss_binomial = entropy_loss_func(pred_sen)
            loss = loss + loss_binomial * self.args.alpha
            log_data[f'{mode}_loss_fairness'] = loss_binomial

        else:
            loss_sen = self.cls_imbalance_loss(pred_sen, label_sen)
            loss = loss_ocean + self.args.beta * loss_sen
            log_data[f'{mode}_loss_sen'] = loss_sen

        self.log_out(log_data, mode)
        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss, 'label_sen': label_sen, 'pred_ocean': pred_ocean, 'label_ocean': label_ocean}
        return ret

    def shared_epoch_end(self, outputs, mode):
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:

            wandb.log({f'{mode}_mse': metric})  # this is the same as the loss_ocean
            log_DIR(outputs, mode)
            log_gap(outputs, mode)
        self.metrics[mode].reset()

    def training_epoch_end(self, outputs):
        mode = 'train'
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            print(f'{mode}_metric: {metric}')
        self.metrics[mode].reset()



