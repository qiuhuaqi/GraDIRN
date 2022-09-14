""" Dataset helper functions """
import random
import numpy as np
from omegaconf.listconfig import ListConfig
import nrrd
import torch
import torch.nn.functional as F
from utils.image import crop_and_pad, normalise_intensity
from utils.image_io import load_nifti


def _to_tensor(data_dict):
    # cast to Pytorch Tensor
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data)
    return data_dict


def _to_ndarry(data_dict):
    # cast to Numpy array
    for name, data in data_dict.items():
        data_dict[name] = data.numpy()
    return data_dict


def _crop_and_pad(data_dict, crop_size):
    # cropping and padding
    for name, data in data_dict.items():
        data_dict[name] = crop_and_pad(data, new_size=crop_size)
    return data_dict


def _normalise_intensity(data_dict, keys, vmin=0., vmax=1.):
    """ Normalise intensity of data in `data_dict` with `keys` """

    # images in one pairing should be normalised using the same min-max
    vmin_in = np.amin(np.array([data_dict[k] for k in keys]))
    vmax_in = np.amax(np.array([data_dict[k] for k in keys]))

    for k, x in data_dict.items():
        if k in keys:
            data_dict[k] = normalise_intensity(x,
                                               min_in=vmin_in, max_in=vmax_in,
                                               min_out=vmin, max_out=vmax,
                                               mode="minmax", clip=True)
    return data_dict


def _resample(data_dict, size=None, scale_factor=None):
    for k, x in data_dict.items():
        if k == 'tar_seg' or k == 'src_seg':
            align_corners = None
            mode = 'nearest'
            x = x.float()
        else:
            align_corners = True
            mode = ['bilinear', 'trilinear'][x.ndim-3]
        data_dict[k] = F.interpolate(x.unsqueeze(0),
                                     size=size, scale_factor=scale_factor,
                                     recompute_scale_factor=True if scale_factor else False,
                                     mode=mode, align_corners=align_corners)[0]
    return data_dict


def _shape_checker(data_dict):
    """Check if all data points have the same shape
    if so return the common shape, if not print data type"""
    data_shapes_dict = {n: x.shape for n, x in data_dict.items()}
    shapes = [x for _, x in data_shapes_dict.items()]
    if all([s == shapes[0] for s in shapes]):
        common_shape = shapes[0]
        return common_shape
    else:
        raise AssertionError(f'Not all data points have the same shape, {data_shapes_dict}')


def _magic_slicer(data_dict, slice_range=None, slicing=None):
    """
    Select all slices, one random slice, or some slices within `slice_range`, according to `slicing`
    Works with ndarray
    """
    # slice selection
    num_slices = _shape_checker(data_dict)[0]

    # set range for slicing
    if slice_range is None:
        # all slices if not specified
        slice_range = (0, num_slices)
    else:
        # check slice_range
        assert isinstance(slice_range, (tuple, list, ListConfig))
        assert len(slice_range) == 2
        assert all(isinstance(s, int) for s in slice_range)
        assert slice_range[0] < slice_range[1]
        assert all(0 <= s <= num_slices for s in slice_range)

    # select slice(s)
    if slicing is None:
        # all slices within slice_range
        slicer = slice(slice_range[0], slice_range[1])

    elif slicing == 'random':
        # randomly choose one slice within range
        z = random.randint(slice_range[0], slice_range[1]-1)
        slicer = slice(z, z + 1)  # use slicer to keep dim

    elif isinstance(slicing, (list, tuple, ListConfig)):
        # slice several slices specified by slicing
        assert all(0 <= i <= 1 for i in slicing), f'Relative slice positions {slicing} need to be within [0, 1]'
        slicer = tuple(int(i * (slice_range[1] - slice_range[0])) + slice_range[0] for i in slicing)

    else:
        raise ValueError(f'Slicing mode {slicing} not recognised.')

    # slicing
    for name, data in data_dict.items():
        data_dict[name] = data[slicer, ...]  # (N, H, W)

    return data_dict


def _clean_seg(data_dict, classes=(0, 1, 2, 3)):
    """ Remove (zero-fill) slices where either ED or ES frame mask is empty """
    # ndarray each of shape
    tar_seg = data_dict['tar_seg']
    src_seg = data_dict['src_seg']
    num_slices = tar_seg.shape[0]
    assert tuple(np.unique(tar_seg)) == tuple(np.unique(src_seg)) == classes
    non_empty_slices = [np.prod([np.sum((tar_seg[i] == cls) * (src_seg[i] == cls)) > 0
                                 for cls in classes]) > 0 for i in range(num_slices)]
    non_empty_slices = np.nonzero(non_empty_slices)[0]
    # slices_to_take = non_empty_slices[[len(non_empty_slices) // 2 + i for i in range(-1, 2, 1)]]
    slices_to_take = non_empty_slices
    tar_seg_out = np.zeros_like(tar_seg)
    src_seg_out = np.zeros_like(src_seg)
    tar_seg_out[slices_to_take] = tar_seg[slices_to_take]
    src_seg_out[slices_to_take] = src_seg[slices_to_take]
    data_dict['tar_seg'] = tar_seg_out
    data_dict['src_seg'] = src_seg_out
    return data_dict


def _load2d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # shape (H, W, N) ->  (N, H, W)
        if 'nrrd' in data_path:
            x, _ = nrrd.read(data_path)
            data_dict[name] = x.transpose(2, 0, 1)
        else:
            data_dict[name] = load_nifti(data_path).transpose(2, 0, 1)
    return data_dict


def _load3d(data_path_dict):
    data_dict = dict()
    for name, data_path in data_path_dict.items():
        # image is saved in shape (H, W, D) -> (ch=1, H, W, D)
        data_dict[name] = load_nifti(data_path)[np.newaxis, ...]
    return data_dict
