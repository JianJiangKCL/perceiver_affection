import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import KnowledgeDistillationLoss


class IncrementTrainer(TrainerABC):
    def __init__(self, teacher_model, student_model, args):
        super(IncrementTrainer, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()
        self.accuracies = {'train': self.train_accuracy, 'val': self.val_accuracy, 'test': self.test_accuracy}
        self.mse_loss = nn.MSELoss()
        self.kd_loss = KnowledgeDistillationLoss(T=20, alpha=0.5)
        self.args = args
        self.root_dir = args.root_dir

    def shared_step(self, batch, mode):
        x, y = batch
        y = y.flatten()
        #todo need to use modality_x
        y_hat = self.student(x)
        with torch.no_grad():
            teacher_output = self.teacher(x)
        mse_loss = self.mse_loss(y_hat, y)
        self.metrics[mode].update(y_hat, y)
        kd_loss = self.kd_loss(y_hat, teacher_output)
        loss = mse_loss + kd_loss

        metric = self.metrics[mode].compute()
        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_kd_loss': kd_loss,
            f'{mode}_mse_loss': mse_loss,
            f'{mode}_metric': metric,
            f'lr': lr
        }
        self.log_dict(log_data, prog_bar=not self.args.disable_tqdm, sync_dist=False if mode == 'train' else True, on_step=True if mode == 'train' else False, on_epoch=False if mode == 'train' else True)

        return loss

    def shared_epoch_end(self, mode):
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            print(f'{mode}_metric: {metric}')
        self.metrics[mode].reset()


