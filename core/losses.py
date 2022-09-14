import math
import numpy as np
import torch
from torch import nn as nn
from core.utils import diff

def ncc(x, y):
    """ Global Normalised Cross Correlation (atm. only for channel=1) """
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)

    x_ = x - x.mean(dim=1, keepdim=True)  # (N, 1)
    y_ = y - y.mean(dim=1, keepdim=True)  # (N, 1)

    with torch.cuda.amp.autocast(enabled=False):  # compute in float32 to avoid overflow
        x_var = x_.square().sum(dim=1)
        y_var = y_.square().sum(dim=1)
        cov2 = (x_ * y_).sum(dim=1).square()
        ncc = cov2 / (x_var * y_var + 1e-5)
    return -ncc.mean()


def diffusion_loss(x: torch.Tensor, spatial_weight: torch.Tensor = None):
    """ L2 regularisation loss"""
    ndims = x.size()[1]
    derives = torch.cat([diff(x, dim=i) for i in range(ndims)], dim=1)
    loss = derives.pow(2).sum(dim=1, keepdim=True)
    if spatial_weight is not None:
        assert loss.size() == spatial_weight.size()
        loss = loss * spatial_weight
    return loss.mean()
