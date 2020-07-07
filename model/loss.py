import numpy as np
import torch
import torch.nn.functional as F

# Classification
def nll_loss(output, target):
    return F.nll_loss(output, target)

# Normal\MC_dropout\BBB\BBB_LR\Qd
def mse_loss(output, target):
    return F.mse_loss(output, target, reduction="sum")

# Normal\MC_dropout\BBB\BBB_LR\Qd
def mse_loss_mean(output, target):
    return F.mse_loss(output, target, reduction="mean")

# DeepEnsemble
def gaussian_nll(output, target):
    mu = output[:, :1]
    sig = output[:, 1:]
    sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
    y_diff = target - mu
    return 0.5 * torch.mean(torch.log(sig_pos)) + 0.5 * torch.mean(
        torch.div(torch.pow(y_diff, 2), sig_pos)) + 0.5 * np.log(2 * np.pi)

# homo
# Normal\MC_dropout\BBB\BBB_LR
def log_gaussian_homo(output, target, sigma, no_dim):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma)
    return - (log_coeff + exponent).sum()

# hetero
# Normal\MC_dropout\BBB\BBB_LR
def log_gaussian_hetero(output, target, no_dim, sum_reduce=True):
    mean = output[:, :1]
    sigma = output[:, 1:].exp()
    exponent = -0.5 * (target - mean) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)
    if sum_reduce:
        return -(log_coeff + exponent).sum()
    else:
        return -(log_coeff + exponent)
