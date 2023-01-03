import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from funcs.utils_funcs import tensor_to_np
import wandb

# knowledge distillation loss
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, T, alpha):
        super(KnowledgeDistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, teacher_output):
        student_output = F.log_softmax(student_output / self.T, dim=1)
        teacher_output = F.softmax(teacher_output / self.T, dim=1)
        return self.kl_div(student_output, teacher_output) * (self.T ** 2) * self.alpha


# fairness loss
OCEAN_MEANS = [0.31256767999999996, 0.3745465626666666, -0.3980745346, -1.47551749, -0.20200107000000006]


# binary OCEAN values in a batch manner
def get_binary_ocean_values(ocean_values, STE=True):
    '''
    #1. Took the mean of each of the OCEAN values from the train set, which gave me the values
    # OPENMINDEDNESS_Z mean is =  0.31256767999999996
    # CONSCIENTIOUSNESS_Z mean is =  0.3745465626666666
    # EXTRAVERSION_Z mean is =  -0.3980745346
    # AGREEABLENESS_Z mean is =  -1.47551749
    # NEGATIVEEMOTIONALITY_Z mean is =  -0.20200107000000006
    #
    # 2. In the test data, if the value of OPENMINDEDNESS_Z of a prediction exceeded that of the mean we got from train set ( the one above ), we took it as positive (1).
    # Else if it was less than the mean above, we took it as 0.

    STE is the straight through gradient estimator, used to pass gradients of binary OCEAN values to the original OCEAN values
    '''
    original_ocean_values = ocean_values
    ocean_values = ocean_values.cpu().detach()#.numpy()
    ocean_values = ocean_values - torch.tensor(OCEAN_MEANS)
    # fast way to get binary OCEAN values
    binary_ocean_values = torch.where(ocean_values > 0, torch.tensor(1), torch.tensor(0))
    # time-consuming version
    # binary_ocean_values = []
    # for i in range(ocean_values.shape[0]):
    #     binary_ocean_values.append([1 if ocean_values[i][0] > OCEAN_MEANS[0] else 0, 1 if ocean_values[i][1] > OCEAN_MEANS[1] else 0,
    #                                 1 if ocean_values[i][2] > OCEAN_MEANS[2] else 0,
    #                                 1 if ocean_values[i][3] > OCEAN_MEANS[3] else 0,
    #                                 1 if ocean_values[i][4] > OCEAN_MEANS[4] else 0])

    binary_ocean_values = torch.tensor(binary_ocean_values)
    if STE:
        # this is derivative.
        ret = original_ocean_values + (binary_ocean_values - original_ocean_values).detach()
    else:
        # this is not derivative.
        ret = binary_ocean_values
    return ret


# DIR is not suitable for mini-batch updating, as the privileged group is not fixed and the three divisions are easy to have 0s.
# preds come in a batch manner, so the size is [batch_size, 5]
# the sensitive_labels is [batch_size]
# DIR close to 1; SPD close to 0
def DIR_metric(OCEAN_bin_preds, sensitive_labels):

    # get indices where sensitive labels are 1 via torch
    indices_1 = torch.where(sensitive_labels == 1)[0]

    # get indices where sensitive labels are 0
    indices_0 = torch.where(sensitive_labels == 0)[0]

    # get the OCEAN values for the indices where sensitive labels are 1
    OCEAN_preds_1 = OCEAN_bin_preds[indices_1]
    # get the OCEAN values for the indices where sensitive labels are 0
    OCEAN_preds_0 = OCEAN_bin_preds[indices_0]
    num_1 = len(indices_1)
    num_0 = len(indices_0)
    if num_1 > num_0:
        privileged_preds = OCEAN_preds_1
        unprivileged_preds = OCEAN_preds_0
        num_privileged = num_1
        num_unprivileged = num_0
    else:
        privileged_preds = OCEAN_preds_0
        unprivileged_preds = OCEAN_preds_1
        num_privileged = num_0
        num_unprivileged = num_1

    num_privileged = torch.tensor(num_privileged).float()
    num_unprivileged = torch.tensor(num_unprivileged).float()
    # iter over the 5 OCEAN features
    DIRs= []
    SPDs = []
    for i in range(5):
        #todo three divisions may have 0s.

        # calculate the proportion of positive predictions (y==1) for the privileged group
        tmp =  torch.sum(privileged_preds[:, i])
        p_privileged = tmp / num_privileged
        # calculate the proportion of positive predictions (y==1) for the unprivileged group
        p_unprivileged = torch.sum(unprivileged_preds[:, i]) / num_unprivileged
        disparate_impact_ratio = p_unprivileged / p_privileged
        DIRs.append(disparate_impact_ratio)
        statistical_parity_difference = p_unprivileged - p_privileged
        SPDs.append(statistical_parity_difference)

    return DIRs, SPDs


def log_DIR(outputs, mode):
    pred_ocean = torch.cat([output['pred_ocean'] for output in outputs])
    binary_pred_ocean = get_binary_ocean_values(pred_ocean, STE=False)
    sensitive_labels = torch.cat([output['label_sen'] for output in outputs])
    # calculate OCEAN individually
    metric_name = ['O', 'C', 'E', 'A', 'N']
    DIRs, SPDs = DIR_metric(binary_pred_ocean, sensitive_labels)
    for i in range(5):
        wandb.log({f'{mode}_DIR_{metric_name[i]}': DIRs[i]})
        wandb.log({f'{mode}_SPD_{metric_name[i]}': SPDs[i]})


# formulation of TPR
# TPR= TP/(TP+FN)
# # formulation of FPR
# FPR= FP/(FP+TN)
# true positive rates (TPR) or false positive rates (FPR)
def compute_xPR(y_pred, y_gt, TPR=True):
    # y_gt = tensor_to_np(y_gt)
    # y_pred = tensor_to_np(y_pred)
    flag = 1 if TPR else 0
    # this implementation may be not derivative.
    xP = torch.sum(torch.logical_and(y_pred == 1, y_gt == flag))
    nxN = torch.sum(torch.logical_and(y_pred == 0, y_gt == flag))
    sum = xP + nxN
    # todo, sum can be 0
    # if sum == 0:
    #     return 1
    return xP / (xP + nxN)


def compute_gap(R1, R0):
    # absolute difference between TPR1 and TPR0
    return np.abs(R1 - R0)

def log_gap(outputs, mode):
    pred_ocean = torch.cat([output['pred_ocean'] for output in outputs])
    binary_pred_ocean = get_binary_ocean_values(pred_ocean, STE=False)
    label_ocean = torch.cat([output['label_ocean'] for output in outputs])
    binary_label_ocean = get_binary_ocean_values(label_ocean, STE=False)

    # calculate OCEAN individually
    metric_name = ['O', 'C', 'E', 'A', 'N']
    for i in range(5):
        R1 = compute_xPR(binary_pred_ocean[:, i], binary_label_ocean[:, i], TPR=True)
        R0 = compute_xPR(binary_pred_ocean[:, i], binary_label_ocean[:, i], TPR=False)
        gap = compute_gap(R1, R0)
        wandb.log(f'{mode}_gap_{metric_name[i]}: {gap}')

# def equal_opportunity_metric():