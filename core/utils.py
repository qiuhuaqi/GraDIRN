import math

import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer


def diff(x, dim, mode="forward", boundary="Neumann"):
    """
    Finite differnece operator

    Args:
        x: (torch.Tensor, shape (N, ndim, *sizes))
        dim: (int) the dimension along which to compute finite difference
        mode: (str) Finite difference direction 'forward', 'backward' or 'central'
        boundary: (str) Boundary handling method, 'Neumann' or 'Dirichlet'
    """
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    # paddings
    paddings = [[0, 0] for _ in range(ndim)]
    if mode == "forward":
        # forward difference: pad after
        paddings[dim][1] = 1
    elif mode == "backward":
        # backward difference: pad before
        paddings[dim][0] = 1
    else:
        raise ValueError(f'Mode {mode} not recognised')

    # reverse and join sublists into a flat list
    # (Pytorch uses last -> first dim order when padding)
    paddings.reverse()
    paddings = [p for ppair in paddings for p in ppair]

    # pad data
    if boundary == "Neumann":
        # Neumann boundary condition
        x_pad = F.pad(x, paddings, mode='replicate')
    elif boundary == "Dirichlet":
        # Dirichlet boundary condition
        x_pad = F.pad(x, paddings, mode='constant')
    else:
        raise ValueError("Boundary condition not recognised.")

    # offset subtract
    x_diff = x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) \
             - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

    return x_diff


def interpolate_nd(x, scale_factor=None, size=None, mode=None):
    """ Wrapper for torch.nn.functional.interpolate """
    if mode == 'nearest':
        mode = mode
        align_corners = None
    else:
        ndims = x.ndim - 2
        align_corners = True
        if ndims == 1:
            mode = 'linear'
        elif ndims == 2:
            mode = 'bilinear'
        elif ndims == 3:
            mode = 'trilinear'
        else:
            raise ValueError(f'Data dimension ({ndims}) must be 2 or 3')
    y = F.interpolate(x,
                      scale_factor=scale_factor,
                      size=size,
                      mode=mode,
                      recompute_scale_factor=False,
                      align_corners=align_corners
                      )
    return y


def create_img_pyramid(x, lvls=1, label=False):
    """ Create image pyramid, low-resolution to high-resolution"""
    x_pyr = [x]
    interp_mode = 'nearest' if label else None
    for l in range(lvls-1):
        x_pyr.append(interpolate_nd(x, scale_factor=0.5 ** (l + 1), mode=interp_mode))
    x_pyr.reverse()  # low resolution to high resolution
    return x_pyr


def reset_scalar(scalar, init=1., min=0, max=1000, requires_grad=True, train_scale=1):
    """ Reset scalar parameters: set attributes, attach projection """
    scalar.data = torch.tensor(init, dtype=scalar.dtype)
    scalar.proj = lambda: scalar.data.clamp_(min, max)
    scalar.requires_grad = requires_grad
    scalar.train_scale = train_scale


def loc_to_idx(locs, size, order='xy'):
    """
    Convert spatial locations to indices in flattened vector (ravel index)
    Formula for x-y order:
    - 2D: idx = x + y * W
    - 3D: idx = x + y * W + z * W * H

    Args:
        locs: a list of coordinates[x, y, (z)] each of shape (N, *size)
        size: (H, W) or (D, H, W)
        order: # todo: implement i-j order (column major)
    Returns:
        idx: (torch.Tensor) index of locs in ravel tensor shape (N, prod(size))
    """
    assert len(locs) == len(size)
    if order == 'ij':
        raise NotImplementedError(f"Not implementetd for order {order}")

    csize = np.cumprod(size[::-1])  # (W, H*W) or (W, H*W, H*W*D)
    idx = locs[0]
    for i, loc in enumerate(locs[1:]):
        idx = idx + loc * csize[i]
    return idx


def tensor_prod(tensor_list):
    """ Element-wise product of a list of tensors """
    out = tensor_list[0]
    for x in tensor_list[1:]:
        out = out * x
    return out


def tensor_sum(tensor_list):
    """ Element-wise sum of a list of tensors """
    out = tensor_list[0]
    for x in tensor_list[1:]:
        out = out + x
    return out


def convNd(ndim,
           in_channels,
           out_channels,
           kernel_size=3,
           stride=1,
           padding=1,
           a=0.):
    """
    Wrapper to instantiate convolution module of generic dimension
    Args:
        in_channels: (int) number of input channels
        out_channels: (int) number of output channels
        kernel_size: (int) size of the convolution kernel
        stride: (int) convolution stride (step size)
        padding: (int) outer padding
        ndim: (int) model dimension
        a: (float) leaky-relu negative slope for He initialisation

    Returns:
        (nn.Module instance) Instance of convolution module of the specified dimension
    """
    assert ndim > 0
    conv_nd = [nn.Conv2d, nn.Conv3d][ndim-2](in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    nn.init.kaiming_uniform_(conv_nd.weight, a=a)
    return conv_nd


class ConvNd(nn.Module):
    def __init__(self, ndim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, a=0.):
        super().__init__()
        assert ndim == 2 or ndim ==3
        self.conv_nd = [nn.Conv2d, nn.Conv3d][ndim-2](in_channels, out_channels, kernel_size,
                                                      stride=stride, padding=padding)
        nn.init.kaiming_uniform_(self.conv_nd.weight, a=a)

    def forward(self, x):
        return self.conv_nd(x)


class BlockAdam(Optimizer):
    r"""Implements Block-Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(BlockAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BlockAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BlockAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Transform the gradient for the second moment
                #print(p.name, hasattr(p, 'reduction_dim'), hasattr(p, 'proj'))
                if hasattr(p, 'reduction_dim'):
                    grad_reduced = torch.sum(grad**2, p.reduction_dim, True)
                else:
                    grad_reduced = grad**2
                exp_avg_sq.mul_(beta2).add_(grad_reduced.mul_(1 - beta2))
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # print('{}: ||grad||_2={}'.format(p.myname, np.sqrt(np.sum(grad.cpu().numpy()**2))))

                # perform the gradient step
                p.data.addcdiv_(-step_size, exp_avg, denom)
                # perform a projection
                if hasattr(p, 'proj'):
                    p.proj()

        return loss