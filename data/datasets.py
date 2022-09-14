import os
import random

from torch.utils.data import Dataset
from data.utils import _load2d, _load3d, _crop_and_pad, _resample, _normalise_intensity, _clean_seg
from data.utils import _magic_slicer, _to_tensor, _to_ndarry


class _BaseDataset(Dataset):
    def __init__(self, data_dir, limit_data: float = 1., batch_size=1):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir
        assert os.path.exists(data_dir), f"Data dir does not exist: {data_dir}"
        self.data_path_dict = dict()
        self.subject_list = self._set_subj_list(limit_data, batch_size)

    def _set_subj_list(self, limit_data, batch_size):
        assert limit_data <= 1., f'Limit data ratio ({limit_data}) must be <= 1 '
        subj_list = sorted(os.listdir(self.data_dir))
        if limit_data < 1.:
            num_subj = len(subj_list)
            subj_list = subj_list[:int(num_subj * limit_data)]  # select the subset
            subj_list *= (int(1 / limit_data) + 1)  # repeat to fill
            subj_list = subj_list[:num_subj]
        return subj_list * batch_size  # normalise by batch size

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir,
                 crop_size=(176, 192, 176),
                 resample_size=(128, 128, 128),
                 limit_data=1.,
                 batch_size=1,
                 modality='t1t1',
                 evaluate=False,
                 atlas_path=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir, limit_data=limit_data, batch_size=batch_size)
        self.crop_size = crop_size
        self.resample_size = resample_size
        self.img_keys = ['tar', 'src']
        self.evaluate = evaluate
        self.modality = modality
        self.atlas_path = atlas_path
        # Note: original data spacings are 1mm in all dimensions
        self.spacing = [rsz / csz for rsz, csz, in zip(self.resample_size, self.crop_size)]

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        if self.evaluate:
            self.src_subj_id = self.subject_list[(index+1) % len(self.subject_list)]
        else:
            self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        # target and source paths
        self.data_path_dict['tar'] = f'{self.tar_subj_path}/T1_brain.nii.gz'
        if self.modality == 't1t1':
            self.data_path_dict['src'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['src'] = f'{self.src_subj_path}/T2_brain.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        if self.evaluate:
            self.img_keys.append('src_ref')
            self.data_path_dict['src_ref'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            self.data_path_dict['tar_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
            self.data_path_dict['src_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)
        if self.crop_size != self.resample_size:
            data_dict = _resample(data_dict, size=tuple(self.resample_size))
        return data_dict


class CardiacMR2D(_BaseDataset):
    def __init__(self,
                 data_dir,
                 limit_data=1.,
                 batch_size=1,
                 slice_range=None,
                 slicing=None,
                 crop_size=(160, 160),
                 spacing=(1.8, 1.8),
                 original_spacing=(1.8, 1.8)
                 ):
        super(CardiacMR2D, self).__init__(data_dir, limit_data=limit_data, batch_size=batch_size)
        self.crop_size = crop_size
        self.img_keys = ['tar', 'src', 'src_ref']
        self.slice_range = slice_range
        self.slicing = slicing
        self.spacing = spacing
        self.original_spacing = original_spacing

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['tar'] = f'{self.subj_path}/sa_ED.nii.gz'
        self.data_path_dict['src'] = f'{self.subj_path}/sa_ES.nii.gz'
        self.data_path_dict['src_ref'] = self.data_path_dict['src']
        self.data_path_dict['tar_seg'] = f'{self.subj_path}/label_sa_ED.nii.gz'
        self.data_path_dict['src_seg'] = f'{self.subj_path}/label_sa_ES.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load2d(self.data_path_dict)
        data_dict = _magic_slicer(data_dict, slice_range=self.slice_range, slicing=self.slicing)
        if self.original_spacing != self.spacing:
            # resample if spacing changes
            data_dict = _to_tensor(data_dict)
            scale_factor = tuple([os / s for (os, s) in zip(self.original_spacing, self.spacing)])
            data_dict = _resample(data_dict, scale_factor=scale_factor)
            data_dict = _to_ndarry(data_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)
        return data_dict


class CardiacMR2D_MM(CardiacMR2D):
    def __init__(self, *args, **kwargs):
        super(CardiacMR2D_MM, self).__init__(*args, **kwargs)

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['tar'] = f'{self.subj_path}/ED_img.nii.gz'
        self.data_path_dict['src'] = f'{self.subj_path}/ES_img.nii.gz'
        self.data_path_dict['src_ref'] = self.data_path_dict['src']
        self.data_path_dict['tar_seg'] = f'{self.subj_path}/ED_seg.nii.gz'
        self.data_path_dict['src_seg'] = f'{self.subj_path}/ES_seg.nii.gz'

    def __getitem__(self, index):
        data_dict = super(CardiacMR2D_MM, self).__getitem__(index)
        if self.slicing != 'random':
            data_dict = _to_tensor(_clean_seg(_to_ndarry(data_dict)))
        return data_dict
