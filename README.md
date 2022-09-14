# [DEPRECATED REPO] see: https://github.com/qiuhuaqi/GraDIRN
# Multi-Resolution Gradient Descent Networks for Image Registration
Code for MICCAI 2022 submission ID 420

Core dependencies:
- Pytorch v1.8.1
- Pytorch Lightning v.1.2.7
- Numpy v1.21.1
- Nibabel v3.1.
- Hydra-core v1.1.0

To run training:
1. Download data and split the data into training and validation. Under train/val directory, organise data for each subjec into one subject directory, with the name of the data files matching the suffix given in the dataloaders in `data/datasets.py`
2. Give the path of your train/val data directory in `conf/train/data/<your_data_config>.yaml` or via command line `data.train_path=<your_train_data_path> data.val_path=<your_val_data_path>`
3. Run `python train.py run_dir=<your_run_dir> data=<your_data_config>` (by default this runs on GPU 0 of your machine with mixed precision.


We use [Hydra](https://hydra.cc/) for configuration parsing, which means all configuration options can be overwritten easily via command line.


More instructions coming soon...
