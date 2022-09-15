#  Embedding Gradient-based Optimization in Image Registration Networks

## Welcome!

This repository contains the code for training, testing and evaluating the method introduced in the paper:   
"Embedding Gradient-based Optimization in Image Registration Networks"  
by Huaqi Qiu, Kerstin Hammernik, Chen Qin, Chen Chen, Daniel Rueckert
 to be presented in [MICCAI 2022](https://conferences.miccai.org/2022/en/)

The paper is available on ArXiv: https://arxiv.org/abs/2112.03915

## Paper TLDR
> (If you have read the paer, you can skip this using the Table of Content in the top-left corner of the README!) 

In this paper, we introduce a new learning-based registration method (clumsily named "GraDIRN" = "Gradient Descent Network for Image Registration")  by connecting the variational energy-based iterative optimization with learning-based iterative networks. Our proposed approach trains ("higher-levle optimization") a DL network that embeds unrolled multiresolution gradient-based energy optimization in its forward pass ("lower-level optimization"), which explicitly enforces image dissimilarity minimization in its update steps.  

![image](https://user-images.githubusercontent.com/17068099/190219234-a8349a8a-f406-4bfd-b257-500e501f6824.png)

Extensive evaluations were performed on registration tasks using 2D cardiac MR and 3D brain MR images. We demonstrate that our approach achieved state-of-the-art registration performance while using fewer learned parameters, 

|2D cardiac MRI intra-subject <br /> (UK biobank, end-diastolic vs. end-systolic) | 3D brain MRI inter-subject <br /> (CamCAN, aging study)|
| :---:   | :---: |
|![cardiac](https://user-images.githubusercontent.com/17068099/190221298-c00c8422-8ff9-47a6-ab73-c0ef16eb643d.png) | ![brain](https://user-images.githubusercontent.com/17068099/190221270-0ac2caef-cf90-44db-933e-8c03df9ef09e.png) |


with good data efficiency and domain robustness: 

| More effectively trained with less data | Robust against domain shift |
| :---:   | :---: |
|![data efficiency](https://user-images.githubusercontent.com/17068099/190224280-cd0731c7-beb8-47b3-a3e7-e0b0672a9173.png) | ![domain shift](https://user-images.githubusercontent.com/17068099/190224295-03ca91db-3ecb-41d9-829b-53a778e4b572.png) |



## Dependencies 
Core dependencies:
- Pytorch v1.8.1
- Pytorch Lightning v.1.2.7
- Numpy v1.21.1
- Nibabel v3.1.
- Hydra-core v1.1.0


## Running the code
### Training
1. Download and organise the data files of each subject into a directory, with the name of the data files matching the suffix given in the dataloaders in `data/datasets.py`
3. Give the path of your train/val data directory in `conf/train/data/<your_data_config>.yaml` or via command line `data.train_path=<your_train_data_path> data.val_path=<your_val_data_path>`
4. Run `python train.py experiment_root=<your_experiment_root_dir> run_dir=<your_run_dir_under_experiment_root>` 

By default this runs on GPU 0 of your machine with mixed precision. Your model should be under `<your_experiment_root_dir>/<your_run_dir_under_experiment_root>`. These behaviours can be changed in `conf/train/conf.yaml`.

> More detailed instructions coming soon...

### Evaluation
> More detailed instructions coming soon...

## Configurations
We use [Hydra](https://hydra.cc/) for configuration parsing, which means all configurations defined in the YAML files can be overwritten easily via command line.
> More detailed instructions coming soon...


## Data
The cardiac MRI data from UK biobank and the CamCAN datasets are publicly available. However, we are not allowed to redistribute the original or the preprocessed data ourselves. The preprocessing process are described in the paper and the original data can be applied via the following links:
- [CamCAN](https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/)
- [UK Biobank](https://www.ukbiobank.ac.uk/enable-your-research)

The data used to evaluate domain shift robustness is freely available from the M&M2 challenge: https://www.ub.edu/mnms-2

## Contact us
Email: [Huaqi Qiu](mailto:hq615@ic.ac.uk)


