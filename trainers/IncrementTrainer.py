import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import KnowledgeDistillationLossCosine #KnowledgeDistillationLoss
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap, FairnessDistributionLoss, entropy_loss_func, SPD_loss
import wandb


class IncrementTrainer(TrainerABC):
    def __init__(self, args, backbone, old_model, modalities, sensitive_groups):
        super(IncrementTrainer, self).__init__(args, backbone, modalities)

        self.cls_imbalance_loss = nn.CrossEntropyLoss()
        self.fairness_loss1 = FairnessDistributionLoss()
        self.fairness_loss = None
        self.train_class_acc = torchmetrics.Accuracy()
        self.val_class_acc = torchmetrics.Accuracy()
        self.test_class_acc = torchmetrics.Accuracy()
        self.classification_metrics = {'train': self.train_class_acc, 'val': self.val_class_acc, 'test': self.test_class_acc}
        self.knowledge_distillation_loss = KnowledgeDistillationLossCosine()
        self.target_sensitive_group = args.target_sensitive_group
        self.sensitive_groups = sensitive_groups
        self.old_model = old_model
        self.old_model.eval()

    def shared_step(self, batch, mode):
        x, label_ocean, label_sen_dict = batch
        label_sen = label_sen_dict[self.target_sensitive_group]
        label_sen = label_sen.long()
        modalities_x = {modality: x[modality] for modality in self.modalities}

        fv = self.backbone.extract_features(modalities_x)

        # don't update the old model
        old_fv = self.old_model.extract_features(modalities_x).detach()

        pred_ocean = self.backbone.to_logits(fv)
        pred_sen = self.backbone.cosine_fc(fv)

        loss_ocean = self.mse_loss(pred_ocean, label_ocean)
        self.metrics[mode].update(pred_ocean, label_ocean)

        log_data = {
            f'{mode}_loss_ocean': loss_ocean,
        }

        loss = loss_ocean

        # loss_binomial = entropy_loss_func(pred_sen)
        # loss = loss + 0.5 * loss_binomial * self.args.alpha
        # log_data[f'{mode}_loss_fairness'] = loss_binomial
        #
        # loss_sen = self.cls_imbalance_loss(pred_sen, label_sen)
        # loss = loss + 0.5 * (1 - self.args.alpha) * loss_sen
        # log_data[f'{mode}_loss_sen'] = loss_sen
        #
        # self.classification_metrics[mode].update(pred_sen, label_sen)
        # class_acc = self.classification_metrics[mode].compute()
        # log_data[f'{mode}_class_acc'] = class_acc

        loss_spd = self.args.gamma * SPD_loss(pred_ocean, label_sen)
        loss = loss + loss_spd

        log_data[f'{mode}_loss_spd'] = loss_spd

        loss_kd = self.knowledge_distillation_loss(fv, old_fv)
        loss = loss + self.args.beta * loss_kd
        log_data[f'{mode}_loss_kd'] = loss_kd

        self.log_out(log_data, mode)
        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
               'label_ocean': label_ocean}
        return ret

    def training_epoch_end(self, outputs):
        mode = 'train'
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            print(f'{mode}_metric: {metric}')
        self.metrics[mode].reset()
        self.classification_metrics[mode].reset()



