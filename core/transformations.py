import itertools
import numpy as np
import torch
from torch.nn import functional as F
from core.utils import tensor_prod, tensor_sum, loc_to_idx


def normalise_disp(disp):
    """
    Spatially normalise displacements to [-1, 1] coordinate system used by Pytorch `grid_sample()`
    Assumes disp size is the same as the corresponding image.

    Args:
        disp: (numpy.ndarray or torch.Tensor, shape (N, ndim, *size)) Displacement field

    Returns:
        disp: (normalised disp)
    """

    ndim = disp.ndim - 2

    if type(disp) is np.ndarray:
        norm_factors = 2. / np.array(disp.shape[2:])
        norm_factors = norm_factors.reshape(1, ndim, *(1,) * ndim)

    elif type(disp) is torch.Tensor:
        norm_factors = torch.tensor(2.) / torch.tensor(disp.size()[2:], dtype=disp.dtype, device=disp.device)
        norm_factors = norm_factors.view(1, ndim, *(1,) * ndim)

    else:
        raise RuntimeError("Input data type not recognised, expect numpy.ndarray or torch.Tensor")
    return disp * norm_factors


def normalise_grid(grid, size):
    # grid shape (N, *size, ndims)
    ndims = len(size)
    scale = 2. / (torch.tensor(size).type_as(grid).view(1, *(1,)*ndims, ndims) - 1)
    grid = grid * scale - 1.
    return grid


def unnormalise_grid(grid, size):
    # grid shape (N, *size, ndims)
    ndims = len(size)
    scale = (torch.tensor(size).type_as(grid).view(1, *(1,)*ndims, ndims) - 1) / 2.
    grid = (grid + 1) * scale
    return grid


def svf_exp(flow, scale=1, steps=5, sampling='bilinear'):
    """ Exponential of velocity field by Scaling and Squaring"""
    disp = flow * (scale / (2 ** steps))
    for _ in range(steps):
        disp = disp + warp(x=disp, disp=disp,
                           interp_mode=sampling)
    return disp


def grid_sample(x, grid):
    """
    N-D grid sample with bi/tri-linear interpolation using Tensor operations to replace the functional `F.grid_sample()`
    Inspired by Adrian Dalca's TensorFlow implementation in `neurite`:
    https://github.com/adalca/neurite/blob/c7bb05d5dae47d2a79e0fe5a8284f30b2304d335/neurite/tf/utils/utils.py#L73

    Args:
        x: (torch.Tensor) shape (N, C, *size)
        grid: (torch.Tensor) shape (N, *size, ndims), locations to sample, normalised, x-y order

    Returns:
        out: x sampled at grid via linear interpolation
    """
    assert x.shape[2:] == grid.shape[1:-1]
    shape = x.shape
    size = shape[2:]
    ndims = len(size)

    grid = unnormalise_grid(grid, size)  # (N, *size, ndims: (x, y, (z)))
    grid_floor = grid.floor()  # (N, *size, ndims: (x0, y0, (z0)))
    grid_ceil = grid_floor + 1  # (N, *size, ndims: (x1, y1, (z1)))

    # calculate offsets (normalised) for interpolation weights
    offset = grid - grid_floor  # (N, *size, ndims: (x-x0, y-y0, (z-z0)))
    offset_inv = 1 - offset  # (N, *size, ndims: (1-(x-x0), 1-(y-y0), (1-(z-z0)))
    offsets = [[offset[..., i] for i in range(ndims)], [offset_inv[..., i] for i in range(ndims)]]

    # border handling (replicate border)
    locs = [[grid_floor[..., i].clamp(0, size[i] - 1) for i in range(ndims)],
            [grid_ceil[..., i].clamp(0, size[i] - 1) for i in range(ndims)]]

    # flatten data for torch.gather()
    x = x.view(*shape[:2], -1)

    out = []
    # iterate over the corners
    for corner_point in itertools.product([0, 1], repeat=ndims):
        # build the weight of the corner
        # if corner_point[i] is 0, take the inverse offset as weight
        # if corner_point[i] is 1, take the offset as weight
        corner_weight = tensor_prod([offsets[1 - corner_point[i]][i] for i in range(ndims)])
        corner_weight = corner_weight.unsqueeze(1)  # (N, 1, *size)

        # find the corner locations
        corner_locs = [locs[corner_point[i]][i].int() for i in range(ndims)]

        # gather the corner values at the corner locations
        corner_idx = loc_to_idx(corner_locs, size, order='xy')
        corner_idx = corner_idx.view(shape[0], 1, -1).repeat(1, shape[1], 1).long()
        corner_val = x.gather(dim=2, index=corner_idx)
        corner_val = corner_val.view(*shape)  # (N, C, *size)

        out_corner = corner_weight * corner_val
        out.append(out_corner)
    out = tensor_sum(out)
    return out


def warp(x, disp, interp_mode="bilinear"):
    """
    Spatially transform an image by sampling at transformed locations (2D and 3D)

    Args:
        x: (Tensor float, shape (N, ch, *sizes)) input image
        disp: (Tensor float, shape (N, ndim, *sizes)) dense disp field in i-j order (NOT spatially normalised to [-1, 1])
        interp_mode: (string) mode of interpolation in grid_sample()

    Returns:
        deformed x, Tensor of the same shape as input
    """
    ndims = x.ndim - 2
    size = x.size()[2:]
    disp = disp.type_as(x)

    # normalise disp to [-1, 1]
    disp = normalise_disp(disp)

    # generate standard mesh grid, i-j order
    grid = torch.meshgrid([torch.linspace(-1, 1, size[i]).type_as(disp) for i in range(ndims)])
    grid = [grid[i].requires_grad_(False) for i in range(ndims)]

    # apply displacements to each direction (broadcasting) (N, *size)
    warped_grid = [grid[i] + disp[:, i, ...] for i in range(ndims)]
    warped_grid = torch.stack(warped_grid[::-1], -1)  # (N, *size, ndims)  # x-y order

    align_corners = None if interp_mode == 'nearest' else True
    return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=align_corners)


def warp_fn(x, disp, grid, interp_mode="bilinear", backwards_grad=False):
    """
    Warping function that takes grid as input (avoids recreating grid)
    disp and grid both shaped: (N, ndims, *size), i-j order
    """
    disp = normalise_disp(disp)
    warped_grid = grid + disp
    ndims = x.ndim - 2

    # change to x-y order for grid_sample functions
    warped_grid = torch.movedim(warped_grid, 1, -1)[..., list(range(ndims))[::-1]]
    align_corners = None if interp_mode == 'nearest' else True
    if backwards_grad:
        assert interp_mode == 'bilinear', 'Only linear interpolation allowed.'
        return grid_sample(x, warped_grid)
    else:
        return F.grid_sample(x, warped_grid, mode=interp_mode, align_corners=align_corners)
