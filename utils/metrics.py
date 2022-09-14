from functools import partial
import numpy as np
import pandas as pd
import scipy.ndimage
import SimpleITK as sitk

from utils.image import bbox_from_mask, bbox_crop
from utils.misc import save_dict_to_csv


def measure_metrics(metric_data, metric_groups, spacing=(1., 1.), dice_volume_2d=True):
    """
    Wrapper function for calculating all metrics
    Args:
        metric_data: (dict) data used for calculation of metrics
        metric_groups: (list of strings) name of metric groups
        spacing: (tuple of floats) physical spacing on each voxel dimension (in mm)
        dice_volume_2d: (bool) if True, measures Dice for 2D cardiac images by volume

    Returns:
        metrics_results: (dict) {metric_name: metric_value}
    """
    # using groups to share some pre-processings
    # keys must match metric_groups and params.metric_groups
    metric_fn_groups = {'disp_metrics': measure_disp_metrics,
                        'image_metrics': measure_image_metrics,
                        'seg_metrics': partial(measure_seg_metrics, spacing=spacing, volume_2d=dice_volume_2d)}

    metric_results = dict()
    for group in metric_groups:
        metric_results.update(metric_fn_groups[group](metric_data))
    return metric_results


def measure_disp_metrics(metric_data):
    """
    Metrics on displacements.
    If roi_mask is given, the disp is masked and only evaluate in the bounding box of the mask.
    """
    disp = metric_data['disp']
    if 'disp_gt' in metric_data.keys():
        disp_gt = metric_data['disp_gt']

    if 'roi_mask' in metric_data.keys():
        # mask the disp if ROI mask is given
        roi_mask = metric_data['roi_mask']  # (N, 1, *(sizes))
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])
        disp = disp * roi_mask
        disp = bbox_crop(disp, mask_bbox)
        if 'disp_gt' in metric_data.keys():
            disp_gt = disp_gt * roi_mask
            disp_gt = bbox_crop(disp_gt, mask_bbox)

    disp_metric_results = dict()

    # Jacobian metrics
    disp_metric_results.update(calculate_jacobian_metrics(disp))

    # direct disp accuracies if ground truth disp is given
    if 'disp_gt' in metric_data.keys():
        disp_metric_results.update({'aee': np.sqrt(((disp - disp_gt) ** 2).sum(axis=1)).mean(),
                                    'rmse_disp': np.sqrt(((disp - disp_gt) ** 2).sum(axis=1).mean())})

    return disp_metric_results


def measure_image_metrics(metric_data):
    """ Metrics comparing images """
    img = metric_data['tar']
    img_pred = metric_data['tar_pred']  # (N, 1, *sizes)

    # crop out image by the roi mask bounding box if given
    if 'roi_mask' in metric_data.keys():
        roi_mask = metric_data['roi_mask']
        mask_bbox, mask_bbox_mask = bbox_from_mask(roi_mask[:, 0, ...])
        img = bbox_crop(img, mask_bbox)
        img_pred = bbox_crop(img_pred, mask_bbox)
    return {'rmse_image': np.sqrt(((img - img_pred) ** 2).mean())}


def measure_seg_metrics(metric_data, spacing=(1., 1., 1.), volume_2d=True):
    """ Metrics comparing segmentations, if `volume_2d=True` Dice is measured by volume for 2D """
    seg_gt = metric_data['tar_seg']
    seg_pred = metric_data['warped_src_seg']

    if seg_gt.ndim == 4 and volume_2d:
        seg_gt_dice = seg_gt.transpose(1, 0, 2, 3)[np.newaxis, ...]
        seg_pred_dice = seg_pred.transpose(1, 0, 2, 3)[np.newaxis, ...]
        dice = multiclass_score(seg_pred_dice, seg_gt_dice, one_class_dice, score_name='dice')
    else:
        dice = multiclass_score(seg_pred, seg_gt, one_class_dice, score_name='dice')

    # meansure surface distance on slices for 2D, volumes for 3D
    asd = multiclass_score(seg_pred, seg_gt, one_class_average_surface_distance, score_name='asd', spacing=spacing)
    hd = multiclass_score(seg_pred, seg_gt, one_class_hausdorff_distance, score_name='hd', spacing=spacing)
    return {**dice, **asd, **hd}


def calculate_jacobian_metrics(disp):
    """
    Calculate Jacobian related regularity metrics.

    Args:
        disp: (numpy.ndarray, shape (N, ndim, *sizes) Displacement field

    Returns:
        folding_ratio: (scalar) Folding ratio (ratio of Jacobian determinant < 0 points)
        mag_grad_jac_det: (scalar) Mean magnitude of the spatial gradient of Jacobian determinant
    """
    negative_det_J = []
    mag_grad_det_J = []
    std_log_det_J = []
    for n in range(disp.shape[0]):
        disp_n = np.moveaxis(disp[n, ...], 0, -1)  # (*sizes, ndim)
        jac_det_n = jacobian_det(disp_n)
        negative_det_J += [(jac_det_n < 0).sum() / np.prod(jac_det_n.shape)]
        mag_grad_det_J += [np.abs(np.gradient(jac_det_n)).mean()]
        std_log_det_J += [np.log(jac_det_n.clip(1e-9, 1e9)).std()]
    return {'negative_det_J': np.mean(negative_det_J),
            'mag_grad_det_J': np.mean(mag_grad_det_J),
            'std_log_det_J': np.mean(std_log_det_J)}


def jacobian_det(disp):
    """
    Calculate Jacobian determinant of displacement field of one image/volume (2D/3D)

    Args:
        disp: (numpy.ndarray, shape (*sizes, ndim)) Displacement field

    Returns:
        jac_det: (numpy.ndarray, shape (*sizes) Point-wise Jacobian determinant
    """
    disp_img = sitk.GetImageFromArray(disp.astype('float32'), isVector=True)
    jac_det_img = sitk.DisplacementFieldJacobianDeterminant(disp_img)
    jac_det = sitk.GetArrayFromImage(jac_det_img)
    return jac_det


def multiclass_score(y_pred, y, score_fn, score_name=None, **kwargs):
    """
    Compute a metric score from label maps over classes

    Args:
        y_pred: (numpy.ndarray, shape (N, 1, *sizes)) predicted label mask
        y: (numpy.ndarray, shape (N, 1, *sizes)) ground truth label mask
        score_fn: function handle of the function that compute the metric for one class
        score_name: name of the score prefixing to the output dict
        kwargs: keyword arguments for score_fn

    Returns:
        score: (dict) {f'{score_name}_class_{label_classes}': class_score, ...}
    """
    label_classes = np.unique(y_pred)[1:]  # excluding background

    scores = {}
    for label_class in label_classes:
        y_pred_class = np.equal(y_pred, label_class).astype(np.float32).squeeze(axis=1)
        y_class = np.equal(y, label_class).astype(np.float32).squeeze(axis=1)
        scores[f'{score_name}_class_{label_class}'] = score_fn(y_pred_class, y_class, **kwargs)
    scores[f'{score_name}_avg'] = np.nanmean([score for score in scores.values()])
    return scores


def one_class_dice(y_pred, y):
    """
    Dice score between two label masks (not one-hot) for one class

    Args:
        y_pred: (numpy.ndarray, shape (N, *sizes)) binary predicted label mask
        y: (numpy.ndarray, shape (N, *sizes)) binary ground truth label mask

    Returns:
        float: Dice score
    """
    true_positive = y_pred * y
    true_positive = true_positive.sum(axis=tuple(range(1, true_positive.ndim)))
    y_pred_positive = y_pred.sum(axis=tuple(range(1, y_pred.ndim)))
    y_positive = y.sum(axis=tuple(range(1, y.ndim)))
    return np.mean(2 * true_positive / (y_pred_positive + y_positive + 1e-7))


def one_class_hausdorff_distance(y_pred, y, spacing=(1., 1.)):
    """
    Hausdorff distance between two label masks (not one-hot) for one class

    Args:
        y_pred: (numpy.ndarray, shape (N, *sizes)) binary predicted label mask
        y: (numpy.ndarray, shape (N, *sizes)) binary ground truth label mask
        spacing (list, float): pixel/voxel spacings

    Returns:
        hausdorff_distance (float)
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    batch_size = y_pred.shape[0]
    result = []

    for i in range(batch_size):
        y_pred_img = sitk.GetImageFromArray(y_pred[i].astype('float32'))
        y_pred_img.SetSpacing(spacing)
        y_img = sitk.GetImageFromArray(y[i].astype('float32'))
        y_img.SetSpacing(spacing)
        try:
            hausdorff_distance_filter.Execute(y_pred_img, y_img)
            hd = hausdorff_distance_filter.GetHausdorffDistance()
            result.append(hd)
        except:
            # skip empty masks
            if y_pred[i].sum() == 0 or y[i].sum() == 0:
                continue
    return np.mean(result)


def one_class_average_surface_distance(y_pred, y, spacing=(1., 1.,)):
    """
    Average surface distance between two label masks (not one-hot) for one class

    Args:
        y_pred: (numpy.ndarray, shape (N, *sizes)) binary predicted label mask
        y: (numpy.ndarray, shape (N, *sizes)) binary ground truth label mask
        spacing (list, float): pixel/voxel spacings

    Returns:
        average surface distance (float)
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    batch_size = y_pred.shape[0]
    result = []

    for i in range(batch_size):
        y_pred_img = sitk.GetImageFromArray(y_pred[i].astype('float32'))
        y_pred_img.SetSpacing(spacing)
        y_img = sitk.GetImageFromArray(y[i].astype('float32'))
        y_img.SetSpacing(spacing)
        try:
            hausdorff_distance_filter.Execute(y_pred_img, y_img)
            hd = hausdorff_distance_filter.GetAverageHausdorffDistance()
        except:
            # skip empty masks
            if y_pred[i].sum() == 0 or y[i].sum() == 0:
                continue
        result.append(hd)
    return np.mean(result)


class MetricReporter(object):
    """
    Collect and report values
        self.collect_value() collects value in `report_data_dict`, which is structured as:
            self.report_data_dict = {'value_name_A': [A1, A2, ...], ... }

        self.summarise() construct the report dictionary if called, which is structured as:
            self.report = {'value_name_A': {'mean': A_mean,
                                            'std': A_std,
                                            'list': [A1, A2, ...]}
                            }
    """
    def __init__(self, id_list, save_dir, save_name='analysis_results'):
        self.id_list = id_list
        self.save_dir = save_dir
        self.save_name = save_name
        self.csv_path = self.save_dir + f'/{self.save_name}.csv'

        self.report_data_dict = {}
        self.report = {}

    def reset(self):
        self.report_data_dict = {}
        self.report = {}

    def collect(self, x):
        for name, value in x.items():
            if name not in self.report_data_dict.keys():
                self.report_data_dict[name] = []
            self.report_data_dict[name].append(value)

    def summarise(self):
        # summarise aggregated results to form the report dict
        for name in self.report_data_dict:
            self.report[name] = {
                'mean': np.mean(self.report_data_dict[name]),
                'std': np.std(self.report_data_dict[name]),
                'list': self.report_data_dict[name]
            }

    def save_mean_std(self):
        report_mean_std = {}
        for metric_name in self.report:
            report_mean_std[metric_name + '_mean'] = self.report[metric_name]['mean']
            report_mean_std[metric_name + '_std'] = self.report[metric_name]['std']
        # save to CSV
        save_dict_to_csv(report_mean_std, self.csv_path)

    def save_df(self):
        # method_column = [str(model_name)] * len(self.id_list)
        # df_dict = {'Method': method_column, 'ID': self.id_list}
        df_dict = {'ID': self.id_list}
        for metric_name in self.report:
            df_dict[metric_name] = self.report[metric_name]['list']

        df = pd.DataFrame(data=df_dict)
        df.to_pickle(self.save_dir + f'/{self.save_name}_df.pkl')