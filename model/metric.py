import torch
import torch.nn.functional as F
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

# Normal\MC_dropout\BBB(BBB_LR)\QD\DE
def mse(output, target, type=None, reduction = "sum"):
    if type in ['MC_dropout','BBB', 'VD']:
        mean = torch.mean(output, dim=2)
        return F.mse_loss(mean, target, reduction=reduction)
    elif type in ['QD']:
        mean = torch.mean(output, dim=1)
        mean = mean.view(len(output), 1)
        return F.mse_loss(mean, target, reduction=reduction)
    elif type in ['DE']:
        return F.mse_loss(output[:, :1], target, reduction=reduction)
    else:
        return F.mse_loss(output, target, reduction=reduction)

# Normal\MC_dropout\BBB(BBB_LR)\QD\DE
def rmse(output, target, type=None):
    return mse(output, target, type, reduction="mean") **0.5

# MC_dropout\BBB(BBB_LR)\QD\DE
def picp(output, target, type=None):
    return picp_mpiw(output, target, type)[0]

def mpiw(output, target, type=None):
    return picp_mpiw(output, target, type)[1]

def picp_mpiw(output, target, type=None):
    if type in ['MC_dropout', 'BBB', 'VD']:
        mean = torch.mean(output, dim=2)
        std = torch.std(output, dim=2)
        y_U = mean + 2 * std
        y_L = mean - 2 * std
    elif type in ['QD']:
        y_U = output[:, :1]
        y_L = output[:, 1:]
    elif type in ['DE']:
        mu = output[:, :1]
        sig = output[:, 1:]
        sig_pos = torch.log(1 + torch.exp(sig)) + 1e-06
        y_U = mu + 2 * sig_pos
        y_L = mu - 2 * sig_pos
    else:
        exit(1)
    zeros = torch.zeros_like(y_U)
    u = torch.max(zeros, torch.sign(y_U - target))
    l = torch.max(zeros, torch.sign(target - y_L))
    picp = torch.mean(torch.mul(u, l))
    mpiw = torch.mean(y_U - y_L)
    return picp, mpiw


# homo
# Normal\MC_dropout\BBB(BBB_LR)
def picp_homo(output, target, type=None):
    return picp_mpiw_homo(output, target, type)[0]

def mpiw_homo(output, target, type=None):
    return picp_mpiw_homo(output, target, type)[1]

def picp_mpiw_homo(output, target, type=None):
    noise = output[1]
    output = output[0]

    if type in ['MC_dropout', 'BBB',  'VD']:
        mean = torch.mean(output, dim=2)
        std = torch.std(output, dim=2)
        total_unc = (noise ** 2 + (2 * std) ** 2) ** 0.5
        y_U = mean + total_unc
        y_L = mean - total_unc
    else:
        y_U = output + noise
        y_L = output - noise
    zeros = torch.zeros_like(y_U)
    u = torch.max(zeros, torch.sign(y_U - target))
    l = torch.max(zeros, torch.sign(target - y_L))
    picp = torch.mean(torch.mul(u, l))
    mpiw = torch.mean(y_U - y_L)
    return picp, mpiw

def mse_homo(output, target, type=None, reduction = "sum"):
    if type in ['MC_dropout','BBB', 'VD']:
        mean = torch.mean(output[0], dim=2)
        return F.mse_loss(mean, target, reduction=reduction)
    else:
        return F.mse_loss(output[0], target, reduction=reduction)

def rmse_homo(output, target, type=None):
    return mse_homo(output, target, type, reduction="mean") **0.5

# hetero
# Normal\MC_dropout\BBB(BBB_LR)
def picp_hetero(output, target, type=None):
    return picp_mpiw_hetero(output, target, type)[0]

def mpiw_hetero(output, target, type=None):
    return picp_mpiw_hetero(output, target, type)[1]

def picp_mpiw_hetero(output, target, type=None):
    if type in ['MC_dropout', 'BBB',  'VD']:
        mean = torch.mean(output[:, :1, :], dim=2)
        std = torch.std(output[:, :1, :], dim=2)
        noise = torch.mean(output[:, 1:, :].exp() ** 2, dim=2)
        total_unc = (noise ** 2 + (2 * std) ** 2) ** 0.5
        y_U = mean + total_unc
        y_L = mean - total_unc
    else:
        noise = output[:, 1:].exp()
        y_U = output[:, :1] + noise
        y_L = output[:, :1] - noise
    zeros = torch.zeros_like(y_U)
    u = torch.max(zeros, torch.sign(y_U - target))
    l = torch.max(zeros, torch.sign(target - y_L))
    picp = torch.mean(torch.mul(u, l))
    mpiw = torch.mean(y_U - y_L)
    return picp, mpiw

def mse_hetero(output, target, type=None, reduction = "sum"):
    if type in ['MC_dropout','BBB', 'VD']:
        mean = torch.mean(output[:, :1, :], dim=2)
        return F.mse_loss(mean, target, reduction=reduction)
    else:
        return F.mse_loss(output[:, :1], target, reduction=reduction)

def rmse_hetero(output, target, type=None):
    return mse_hetero(output, target, type, reduction="mean") **0.5