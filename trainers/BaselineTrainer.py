import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import get_binary_ocean_values, DIR_metric


class BaselineTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities):
        super(BaselineTrainer, self).__init__(args, backbone=backbone, modalities=modalities)
        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()
        self.metrics = {'train': self.train_metric, 'val': self.val_metric, 'test': self.test_metric}
        # define a regression loss
        self.criterion = nn.MSELoss()

    def shared_step(self, batch, mode):
        x, label_ocean = batch
        modalities_x = {modality: x[modality] for modality in self.modalities}
        
        pred_ocean = self.backbone(modalities_x)
        loss = self.criterion(pred_ocean, label_ocean)
        self.metrics[mode].update(pred_ocean, label_ocean)
        metric = self.metrics[mode].compute()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_metric': metric,
            f'lr': lr
        }
        
        self.log_dict(log_data, prog_bar=not self.args.disable_tqdm, sync_dist=False if mode == 'train' else True, on_step=True if mode == 'train' else False, on_epoch=False if mode == 'train' else True)
        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss}

        return ret

    def shared_epoch_end(self, outputs, mode):
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            print(f'{mode}_metric: {metric}')

        self.metrics[mode].reset()


