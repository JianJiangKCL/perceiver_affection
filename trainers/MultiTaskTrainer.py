import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import KnowledgeDistillationLoss
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap, FairnessDistributionLoss, entropy_loss_func, SPD_loss
import wandb
from einops import rearrange

class MultiTaskTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities, sensitive_groups):
        super(MultiTaskTrainer, self).__init__(args, backbone, modalities)
        self.train_metric = torchmetrics.MeanSquaredError()
        
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()
        self.metrics = {'train': self.train_metric, 'val': self.val_metric, 'test': self.test_metric}
        self.train_class_acc = torchmetrics.Accuracy()
        self.val_class_acc = torchmetrics.Accuracy()
        self.test_class_acc = torchmetrics.Accuracy()
        self.classification_metrics = {'train': self.train_class_acc, 'val': self.val_class_acc, 'test': self.test_class_acc}
        self.mse_loss = nn.MSELoss()
        self.cls_imbalance_loss = nn.CrossEntropyLoss()
        self.fairness_loss1 = FairnessDistributionLoss()
        self.fairness_loss = None
        self.target_sensitive_group = args.target_sensitive_group
        self.sensitive_groups = sensitive_groups

    def shared_step(self, batch, mode):
        x, label_ocean, label_sen_dict = batch
        label_sen = label_sen_dict[self.target_sensitive_group]
        label_sen = label_sen.long()
        if self.args.arch == 'perceiver':
            modalities_x = {modality: x[modality] for modality in self.modalities}

            fv = self.backbone.extract_features(modalities_x)
            pred_ocean = self.backbone.to_logits(fv)
            # pred_sen = self.backbone.cosine_fc(fv)

            loss_ocean = self.mse_loss(pred_ocean, label_ocean)
        elif self.args.arch == 'infomax':
            modalities_x = {modality: rearrange(x[modality], 'b d () -> b  d') for modality in self.modalities}
            lld, nce, pred_ocean, pn_dic, H, _ = self.backbone(modalities_x)
            # alpha defaut 0.3; sigma default 0.1
            loss_ocean = self.mse_loss(pred_ocean, label_ocean) + self.args.alpha * nce - self.args.sigma * lld

        self.metrics[mode].update(pred_ocean, label_ocean)

        log_data = {
            f'{mode}_loss_ocean': loss_ocean,
        }

        loss = loss_ocean


        loss_spd = self.args.gamma * SPD_loss(pred_ocean, label_sen, three_way=True if self.args.target_sensitive_group=='ethnicity' else False)
        loss = loss + loss_spd

        log_data[f'{mode}_loss_spd'] = loss_spd


        self.log_out(log_data, mode)
        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean, 'label_ocean': label_ocean}
        return ret



    def training_epoch_end(self, outputs):
        mode = 'train'
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            print(f'{mode}_metric: {metric}')
        self.metrics[mode].reset()
        self.classification_metrics[mode].reset()



