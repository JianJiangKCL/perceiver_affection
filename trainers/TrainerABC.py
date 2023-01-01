import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from funcs.module_funcs import setup_optimizer, setup_scheduler
from models.perceiver import Perceiver


class TrainerABC(pl.LightningModule):
    args: AttributeDict
    backbone: Perceiver
    def __init__(self, args, backbone=None, modalities=None):
        super(TrainerABC, self).__init__()
        self.train_metric = None
        self.val_metric = None
        self.test_metric = None
        self.metrics = None
        # define a regression loss
        self.criterion = None
        self.modalities = sorted(modalities)
        self.args = args
        self.backbone = backbone

    def configure_optimizers(self):
        opt = setup_optimizer(self.args, self.backbone)
        scheduler = setup_scheduler(self.args, opt, milestones=self.args.milestones)
        if scheduler is None:
            return opt
        return [opt], [scheduler]

    def shared_step(self, batch, mode):
        loss = None
        ret = {'loss': loss}
        # the returns are dict so cannot be stacked directly
        return ret

    def shared_epoch_end(self, outputs, mode):
        pass

    def training_step(self, batch, batch_idx):
        ret = self.shared_step(batch, 'train')

        # must return loss
        return ret

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        ret = self.shared_step(batch, 'val')
        return ret

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        ret = self.shared_step(batch, 'test')
        return ret

    def test_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'test')

    def log_out(self, log_data, mode):
        if mode == 'train':
            log_data['lr'] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict(log_data, prog_bar=not self.args.disable_tqdm, sync_dist=False if mode == 'train' else True,
                      on_step=True if mode == 'train' else False, on_epoch=False if mode == 'train' else True)

