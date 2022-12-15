import pytorch_lightning as pl
import torch.nn as nn

from pytorch_lightning.utilities import AttributeDict
import torchmetrics
import os
from torchvision.models import resnet18, mobilenet_v2, vgg16_bn
from funcs.module_funcs import setup_optimizer, setup_scheduler


class BaselineTrainer(pl.LightningModule):
    args: AttributeDict
    def __init__(self, args, backbone=None, modalities=None):
        super(BaselineTrainer, self).__init__()
        self.metric_sum = 0
        self.n_sum = 0
        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()
        self.metrics = {'train': self.train_metric, 'val': self.val_metric, 'test': self.test_metric}
        # define a regression loss
        self.criterion = nn.MSELoss()
        self.modalities = sorted(modalities)
        self.args = args
        self.backbone = backbone

    def configure_optimizers(self):
        opt = setup_optimizer(self.args, self.backbone)
        scheduler = setup_scheduler(self.args, opt)
        return [opt], [scheduler]

    def shared_step(self, batch, mode):
        x, y = batch
        modalities_x = {modality: x[modality] for modality in self.modalities}
        
        y_hat = self.backbone(modalities_x)
        loss = self.criterion(y_hat, y)
        self.metrics[mode].update(y_hat, y)
        metric = self.metrics[mode].compute()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_metric': metric,
            f'lr': lr
        }
        self.log_dict(log_data, prog_bar=not self.args.disable_tqdm, sync_dist=False if mode == 'train' else True)

        return loss

    def shared_epoch_end(self, mode):
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            print(f'{mode}_metric: {metric}')
        self.metrics[mode].reset()

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'train')
        # must return loss
        return loss

    def training_epoch_end(self, outputs):
        self.shared_epoch_end('train')

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end('val')

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, 'test')

    def test_epoch_end(self, outputs):
        self.shared_epoch_end('test')

