import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap
import wandb
from einops import rearrange


class PreProcTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities, sensitive_groups):
        super(PreProcTrainer, self).__init__(args, backbone=backbone, modalities=modalities)
        self.target_sensitive_group = args.target_sensitive_group
        self.sensitive_groups = sensitive_groups
        #

        # reduction='none' is important for the sample_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        print('PreProcTrainer')

    def shared_step(self, batch, mode):
        if mode == 'train':
            x, label_ocean, label_sen_dict, sample_weights = batch

        else:
            x, label_ocean, label_sen_dict = batch
            sample_weights = torch.ones_like(label_ocean[:, 0])

        modalities_x = {modality: x[modality] for modality in self.modalities}
        pred_ocean = self.backbone(modalities_x)

        if self.args.target_personality is not None:
            if self.args.num_outputs == 5:
                loss_mse = self.mse_loss(pred_ocean[:, self.args.target_personality],
                                         label_ocean[:, self.args.target_personality])
                self.metrics[mode].update(pred_ocean[:, self.args.target_personality],
                                          label_ocean[:, self.args.target_personality])
            elif self.args.num_outputs == 1:
                loss_mse = self.mse_loss(pred_ocean,
                                         label_ocean[:, self.args.target_personality])
                self.metrics[mode].update(pred_ocean,
                                          label_ocean[:, self.args.target_personality].unsqueeze(1))
                k = 1

        else:
            loss_mse = self.mse_loss(pred_ocean, label_ocean)
            self.metrics[mode].update(pred_ocean, label_ocean)

        loss = loss_mse * sample_weights
        loss = loss.mean()


        metric = self.metrics[mode].compute()
        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_metric': metric,
        }

        self.log_out(log_data, mode)

        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
               'label_ocean': label_ocean}  # , 'mse_wo_reduction': mse_wo_reduction}

        return ret

    # def shared_epoch_end(self, outputs, mode):
    #     local_rank = os.getenv("LOCAL_RANK", 0)
    #     metric = self.metrics[mode].compute()
    #     if local_rank == 0:
    #         wandb.log({f'{mode}_mse': metric})
    #         log_DIR(outputs, mode)
    #         log_gap(outputs, mode)
    #
    #     self.metrics[mode].reset()


