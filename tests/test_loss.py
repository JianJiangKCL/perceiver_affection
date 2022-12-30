import torch
from models.losses import compute_xPR
import numpy as np
from funcs.utils_funcs import tensor_to_np
######
# mean squared error loss
def test_mse_loss():
	x1 = torch.randn( 5, 2)
	x2 = torch.randn( 5, 2)

	mean_squared_error = torch.nn.MSELoss(reduction='mean')

	error1 = mean_squared_error(x1, x2)

	sum = 0
	for i in range( 5):
		sum += (x1[i] - x2[i])** 2

	error2 = sum /  5
########
#########
# Test xPR
def test_xPR():
	y = torch.tensor(   [0, 0, 0, 1, 1])
	# y_gt = torch.tensor([0, 1, 0, 1, 1])
	y_gt = torch.tensor([0, 0, 0, 0, 0])
	# calculate TP, FP, FN, TN
	TP = torch.sum(y_gt * y)
	FP = torch.sum(y * ( 1 - y_gt))
	FN = torch.sum(( 1 - y) * y_gt)
	TN = torch.sum(( 1 - y) * ( 1 - y_gt))
	TPR1 = TP / (TP + FN)
	FPR1 = FP / (FP + TN)
	y = tensor_to_np(y)
	y_gt = tensor_to_np(y_gt)
	tmp = np.sum(np.logical_and(y == 0, y_gt ==  0))
	# True Positive rate
	TPR2 = compute_xPR(y, y_gt, TPR=True)
	# False Positive rate
	FPR = compute_xPR(y, y_gt, TPR=False)
#########


def test_binary_OCEAN():
	pass