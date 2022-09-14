#  Embedding Gradient-based Optimization in Image Registration Networks

## Welcome!

This repository contains the code for training, testing and evaluating the method introduced in the paper:   
"Embedding Gradient-based Optimization in Image Registration Networks"  
by Huaqi Qiu, Kerstin Hammernik, Chen Qin, Chen Chen, Daniel Rueckert
 to be presented in [MICCAI 2022](https://conferences.miccai.org/2022/en/)

The paper is available on ArXiv: https://arxiv.org/abs/2112.03915

## Paper TLDR
In this paper, we introduce a new learning-based registration method by connecting the variational energy-based iterative optimization with learning-based iterative networks. Our proposed approach trains a DL network that embeds unrolled multiresolution gradient-based energy optimization in its forward pass, which explicitly enforces image dissimilarity minimization in its update steps.  

![image](https://user-images.githubusercontent.com/17068099/190219234-a8349a8a-f406-4bfd-b257-500e501f6824.png)

Extensive evaluations were performed on registration tasks using 2D cardiac MR and 3D brain MR images. We demonstrate that our approach achieved state-of-the-art registration performance while using fewer learned parameters, with good data efficiency and domain robustness.

<p align="center">
  <img 
       alt="cardiac" 
       src="https://user-images.githubusercontent.com/17068099/190221298-c00c8422-8ff9-47a6-ab73-c0ef16eb643d.png" 
       title="2D cardiac MRI intra-subject registration (UK biobank, end-diastolic vs. end-systolic)" 
       width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="brain" 
       src="https://user-images.githubusercontent.com/17068099/190221270-0ac2caef-cf90-44db-933e-8c03df9ef09e.png" 
       width="45%"
       title="3D brain MRI inter-subject registration (CamCAN, aging study)">
</p>


<p align="center">
  <img alt="cardiac" src="https://user-images.githubusercontent.com/17068099/190224280-cd0731c7-beb8-47b3-a3e7-e0b0672a9173.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="brain" src="https://user-images.githubusercontent.com/17068099/190224295-03ca91db-3ecb-41d9-829b-53a778e4b572.png" width="45%">
</p>



## Dependencies 
Core dependencies:
- Pytorch v1.8.1
- Pytorch Lightning v.1.2.7
- Numpy v1.21.1
- Nibabel v3.1.
- Hydra-core v1.1.0


## Running the code
To run training:
1. Download data and split the data into training and validation. Under train/val directory, organise data for each subject into one subject directory, with the name of the data files matching the suffix given in the dataloaders in `data/datasets.py`
2. Give the path of your train/val data directory in `conf/train/data/<your_data_config>.yaml` or via command line `data.train_path=<your_train_data_path> data.val_path=<your_val_data_path>`
3. Run `python train.py run_dir=<your_run_dir> data=<your_data_config>` (by default this runs on GPU 0 of your machine with mixed precision.

## Configurations
We use [Hydra](https://hydra.cc/) for configuration parsing, which means all configuration options can be overwritten easily via command line.


More instructions coming soon...
