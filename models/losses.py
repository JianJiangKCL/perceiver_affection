import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from funcs.utils_funcs import tensor_to_np
import wandb
from einops import rearrange, repeat


global OCEAN_MEANS


def set_ocean_means(ocean_means):
    global OCEAN_MEANS
    OCEAN_MEANS = ocean_means


# knowledge distillation loss
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, T):
        # alpha is the weight of the loss
        # T is the temperature
        super(KnowledgeDistillationLoss, self).__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, teacher_output):
        student_output = F.log_softmax(student_output / self.T, dim=1)
        teacher_output = F.softmax(teacher_output / self.T, dim=1)
        return self.kl_div(student_output, teacher_output) * (self.T ** 2)


# fairness loss
# OCEAN_MEANS = [0.31256767999999996, 0.3745465626666666, -0.3980745346, -1.47551749, -0.20200107000000006]


# kl divergence between predicted distribution and binomial distribution
class FairnessDistributionLoss(nn.Module):
    def __init__(self, T=1):
        super(FairnessDistributionLoss, self).__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

        binomial_dist = torch.distributions.Binomial(1, 0.5)
        # [0.5, 0.5]
        dist_prob = binomial_dist.log_prob(torch.arange(2)).exp()
        self.dist_prob = dist_prob.cuda()

    def forward(self, output):
        output = F.softmax(output / self.T, dim=1)

        # dist_prob = dist_prob / self.T # the temperature is not needed here
        dist_prob = repeat(self.dist_prob, 'n -> b n', b=output.shape[0])
        return self.kl_div(output, dist_prob) * (self.T ** 2)


def entropy_loss_func(logits):
    def clamp_probs(probs):
        # avoid 0 probs to cause nan
        eps = torch.finfo(probs.dtype).eps
        return probs.clamp(min=eps, max=1 - eps)

    probs = F.softmax(logits, dim=1)
    probs_clamped = clamp_probs(probs)
    logits = torch.log(probs_clamped)

    p_log_p = logits * probs
    entropy = -p_log_p.sum(-1)
    return entropy.mean()


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


def separate_binary_label_group(to_separate, labels):
    # get indices where labels are 1 via torch
    indices_1 = torch.where(labels == 1)[0]

    # get indices where sensitive labels are 0
    indices_0 = torch.where(labels == 0)[0]

    # get the OCEAN values for the indices where labels are 1
    group_1 = to_separate[indices_1]
    # get the OCEAN values for the indices where labels are 0
    group_0 = to_separate[indices_0]

    return group_0, group_1


# DIR is not suitable for mini-batch updating, as the privileged group is not fixed and the three divisions are easy to have 0s.
# even in the batch manner, the DIR can have 0s.
# preds come in a batch manner, so the size is [batch_size, 5]
# the sensitive_labels is [batch_size]
# DIR close to 1; SPD close to 0
def DIR_metric(OCEAN_bin_preds, sensitive_labels):

    OCEAN_preds_0, OCEAN_preds_1 = separate_binary_label_group(OCEAN_bin_preds, sensitive_labels)
    num_1 = len(OCEAN_preds_1)
    num_0 = len(OCEAN_preds_0)
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

        # calculate the proportion of positive predictions (y==1) for the privileged group

        if num_privileged != 0 and num_unprivileged != 0:
            p_privileged = torch.sum(privileged_preds[:, i]) / num_privileged
            # calculate the proportion of positive predictions (y==1) for the unprivileged group
            p_unprivileged = torch.sum(unprivileged_preds[:, i]) / num_unprivileged
            if p_privileged == 0:
                p_privileged = 0.0001
            disparate_impact_ratio = p_unprivileged / p_privileged

            DIRs.append(disparate_impact_ratio)
            statistical_parity_difference = p_unprivileged - p_privileged
            SPDs.append(statistical_parity_difference)
        elif num_privileged == 0 and num_unprivileged != 0:
            # p_privileged to infinity, DIR to 0, SPD to minus infinity
            DIRs.append(0)
            SPDs.append(-99)

        elif num_privileged != 0 and num_unprivileged == 0:
            # p_unprivileged to infinity, DIR to infinity, SPD to infinity
            DIRs.append(99)
            SPDs.append(99)

    return DIRs, SPDs


def log_DIR(outputs, sensitive_group, mode):
    pred_ocean = torch.cat([output['pred_ocean'] for output in outputs])
    binary_pred_ocean = get_binary_ocean_values(pred_ocean, STE=False)

    sensitive_labels = torch.cat([output['label_sen_dict'][sensitive_group] for output in outputs])

    # calculate OCEAN individually
    metric_name = ['O', 'C', 'E', 'A', 'N']
    DIRs, SPDs = DIR_metric(binary_pred_ocean, sensitive_labels)
    for i in range(5):
        wandb.log({f'{sensitive_group}_{mode}_DIR_{metric_name[i]}': DIRs[i]})
        wandb.log({f'{sensitive_group}_{mode}_SPD_{metric_name[i]}': SPDs[i]})


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


def log_gap(outputs, sensitive_group, mode, TPR=True):
    pred_ocean = torch.cat([output['pred_ocean'] for output in outputs])
    binary_pred_ocean = get_binary_ocean_values(pred_ocean, STE=False)
    label_ocean = torch.cat([output['label_ocean'] for output in outputs])
    binary_label_ocean = get_binary_ocean_values(label_ocean, STE=False)
    sensitive_labels = torch.cat([output['label_sen_dict'][sensitive_group] for output in outputs])
    
    OCEAN_preds_0, OCEAN_preds_1 = separate_binary_label_group(binary_pred_ocean, sensitive_labels)

    label_ocean_0, label_ocean_1 = separate_binary_label_group(binary_label_ocean, sensitive_labels)

    # calculate OCEAN individually
    for TPR in [True, False]:
        prefix = 'TPR' if TPR else 'FPR'
        metric_name = ['O', 'C', 'E', 'A', 'N']
        for i in range(5):
            xPR_0 = compute_xPR(OCEAN_preds_0[:, i], label_ocean_0[:, i], TPR=TPR)
            xPR_1 = compute_xPR(OCEAN_preds_1[:, i], label_ocean_1[:, i], TPR=TPR)
            gap = compute_gap(xPR_0, xPR_1)
            wandb.log({f'{sensitive_group}_{mode}_{prefix}gap_{metric_name[i]}': gap})

# def equal_opportunity_metric():
