"""Run model inference and save outputs for analysis"""
import os
import time
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datasets import BrainMRInterSubj3D, CardiacMR2D, CardiacMR2D_MM
from models.gradirns import LitGraDIRN
from models.base import LitDLReg
from core.transformations import warp
from core.utils import create_img_pyramid
from utils.image_io import save_nifti, load_nifti
from utils.metrics import measure_metrics, MetricReporter
from utils.misc import setup_dir

import random
random.seed(7)

DATASETS = {'brain_camcan': BrainMRInterSubj3D, 'cardiac_ukbb': CardiacMR2D, 'cardiac_mm': CardiacMR2D_MM}
DL_MODELS = {'voxelmorph': LitDLReg, 'gradirn': LitGraDIRN}


def get_test_dataloader(cfg, pin_memory=False):
    dataset = DATASETS[cfg.data.type](**cfg.data.dataset)
    return DataLoader(dataset,
                      shuffle=False,
                      pin_memory=pin_memory,
                      **cfg.data.dataloader)


def get_test_model(cfg, device=torch.device('cpu')):
    if cfg.model.type in DL_MODELS.keys():
        model = DL_MODELS[cfg.model.type].load_from_checkpoint(cfg.model.ckpt_path, strict=True)
        model = model.to(device=device)
        model.eval()
    else:
        raise ValueError(f"Unknown inference model type: {cfg.model.type}")
    return model


def inference(model, dataloader, output_dir, model_type=None, device=torch.device('cpu')):
    print('---------------------')
    print("Running inference...")

    for idx, batch in enumerate(tqdm(dataloader)):
        for k, x in batch.items():
            # reshape data for inference
            # 2d: (batch_size=1, num_slices, H, W) -> (num_slices, batch_size=1, H, W)
            # 3d: (batch_size=1, 1, D, H, W) -> (1, batch_size=1, D, H, W) only works with batch_size=1
            batch[k] = x.transpose(0, 1).to(device=device)

        # model inference
        with torch.no_grad():
            if model_type == 'gradirn':
                num_resolutions = len(model.hparams.network.config.num_blocks)
                tars = create_img_pyramid(batch['tar'], num_resolutions)
                srcs = create_img_pyramid(batch['src'], num_resolutions)

                out = model(tars, srcs)
            else:
                out = model(batch['tar'], batch['src'])
            disp = out['disp'][-1].cpu().numpy()

        # save the outputs
        subj_id = dataloader.dataset.subject_list[idx]
        output_subj_dir = setup_dir(f'{output_dir}/{subj_id}')
        # reshape for saving (for visualising with external tools):
        # 2D: (N=num_slice, 2, H, W) -> (H, W, N, 2)
        # 3D: (N=1, 3, D, H, W) -> (D, H, W, 3)
        disp = np.moveaxis(disp, [0, 1], [-2, -1]).squeeze()
        save_nifti(disp, path=f'{output_subj_dir}/disp.nii.gz')


def analyse(test_dataloader, inference_output_dir, save_dir, metric_groups, data_type):
    print('---------------------')
    print("Running analysis...")
    assert len(os.listdir(inference_output_dir)) > 0, "Run inference first!"

    metric_reporter = MetricReporter(id_list=test_dataloader.dataset.subject_list, save_dir=save_dir)

    for idx, batch in enumerate(tqdm(test_dataloader)):
        for k, x in batch.items():
            # reshape data
            # 2d: (batch_size=1, num_slices, H, W) -> (num_slices, batch_size=1, H, W)
            # 3d: (batch_size=1, 1, D, H, W) -> (1, batch_size=1, D, H, W) only works with batch_size=1
            batch[k] = x.transpose(0, 1)

        # load inference output disp
        subj_id = test_dataloader.dataset.subject_list[idx]
        disp = load_nifti(f'{inference_output_dir}/{subj_id}/disp.nii.gz')
        ndims = disp.shape[-1]
        if ndims == 2:  # (H, W, N, 2) -> (N=num_slice, 2, H, W)
            disp = disp.transpose(2, 3, 0, 1)
        if ndims == 3:  # (D, H, W, 3) -> (N=1, 3, D, H, W)
            disp = disp.transpose(3, 0, 1, 2)[np.newaxis, ...]
        disp = torch.from_numpy(disp)
        batch['disp'] = disp

        # warp images and segmentation using predicted disp
        batch['warped_src'] = warp(batch['src'], disp)
        batch['warped_src_seg'] = warp(batch['src_seg'], disp, interp_mode='nearest')
        batch['tar_pred'] = warp(batch['src_ref'], disp)

        # calculate metric for one validation batch
        metric_data = {k: x.numpy() for k, x in batch.items()}
        metric_result_step = measure_metrics(metric_data, metric_groups, spacing=test_dataloader.dataset.spacing)
        metric_reporter.collect(metric_result_step)

    # save the metric results
    metric_reporter.summarise()
    metric_reporter.save_mean_std()
    metric_reporter.save_df()

    # print mean and std metrics
    df = pd.read_csv(metric_reporter.csv_path)
    pd.set_option("max_rows", None)
    pd.set_option("max_columns", None)
    print(df.head())


@hydra.main(config_path="conf/test", config_name="config")
def main(cfg: DictConfig) -> None:

    # configure GPU
    if isinstance(cfg.gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("=============================")
    print("Device: ", device)
    print("=============================")

    # configure dataset & model
    test_dataloader = get_test_dataloader(cfg, pin_memory=(device is torch.device('cuda')))

    # run inference
    run_dir = HydraConfig.get().run.dir

    # run inference
    output_dir = setup_dir(f'{run_dir}/outputs')
    if cfg.inference:
        test_model = get_test_model(cfg, device=device)
        inference(test_model, test_dataloader, output_dir, model_type=cfg.model.type, device=device)

    # run analysis on the inference outputs
    analysis_dir = setup_dir(f'{run_dir}/analysis')
    if cfg.analyse:
        analyse(test_dataloader, output_dir, analysis_dir, cfg.metric_groups, cfg.data.type)


if __name__ == '__main__':
    main()
