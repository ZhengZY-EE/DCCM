## DCCM: Dual Data Consistency Guided Consistency Model For Inverse Problems <br><sub>Official implementation of the ICIP 2024 paper (Oral, Top 5% of the accpeted papers)

**DCCM: Dual Data Consistency Guided Consistency Model For Inverse Problems** <br>
Jiahao Tian, Ziyang Zheng, Xinyu Peng, Yong Li, Wenrui Dai, Hongkai Xiong
https://ieeexplore.ieee.org/abstract/document/10647264/

**Abstract:** *Existing diffusion models for inverse problems have demonstrated impressive performance but suffer from prohibitive sampling complexity due to lengthy iterative sampling procedures. While introducing pre-trained consistency models (CMs) as priors holds promise for fast and high-quality sampling, theoretical disparities between CMs and diffusion models remain, hindering the application of CMs in solving inverse problems. To address this issue, we propose a novel framework, named Dual Data Consistency Guided Consistency Model (DCCM), that for the first time to solve inverse problems with pretrained CM priors. We establish a denoising interpretation of CMs to set up the equivalence between CMs and denoisers and incorporate CM in a theoretically sound fashion. Consequently, we develop refined data consistency to facilitate optimization with CM priors and avoid local minima caused by the nonlinearity of degradation operators. Furthermore, we introduce the data consistency shortcut that leverages the manifold hypothesis to approximate refined data consistency and bypass backpropagation for enhanced sampling speed without reconstruction quality loss. Extensive experiments demonstrate DCCM achieves state-of-the-art performance in terms of reconstruction quality and sampling speed in a wide range tasks of image deblurring, super-resolution, and inpainting.*


### Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1

### Getting started 

#### 1) Clone the repository

#### 2) Download pretrained checkpoint
From the [link](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt), download the pretrained CM on LSUN Bedroom-256.
From the [link](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt), download the pretrained CM on LSUN Cat-256.

Paste the models to  ```./models```

#### 3) Set environment
We follow the environment settings in [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) and [Consistency Model](https://github.com/openai/consistency_models).

Use the external codes for motion-blurring and non-linear deblurring (Igonore if you have downloaded the codes include these external codes). Note that ```GOPRO_wVAE.pth``` shoud be downloaded and placed in ```bkse/\experiments\pretrained```.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
git clone https://github.com/LeviBorodenko/motionblur motionblur
```

Install dependencies

```
conda create -n DCCM python=3.8
conda activate DCCM
pip install -e .
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

#### 4) Inference
We provide example inference codes in ```./run```.
Choose the type of inverse problem you want to solve, e.g. gaussian deblur. We can start inference as follows:
```
bash run/bed_gaussiandeblur.sh
```
##### Possible task configurations
You can change the configs for different inverse problems in ```./configs```

Supported inverse problems:
```
# Linear inverse problems
- configs/super_resolution_config.yaml
- configs/gaussian_deblur_config.yaml
- configs/motion_deblur_config.yaml
- configs/inpainting_config.yaml
- configs/inpainting_config_box.yaml

# Non-linear inverse problems
- configs/nonlinear_deblur.yaml
- configs/phase_retrieval_config.yaml
- configs/hdr_config.yaml
```
### Citation
If you find our work interesting, please consider citing
```
@inproceedings{tian2024dccm,
  title={DCCM: Dual Data Consistency Guided Consistency Model for Inverse Problems},
  author={Tian, Jiahao and Zheng, Ziyang and Peng, Xinyu and Li, Yong and Dai, Wenrui and Xiong, Hongkai},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)},
  pages={1507--1513},
  year={2024},
  organization={IEEE}
}
```

