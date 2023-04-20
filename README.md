**News**

* `02/12/2023` We release ENeRF object-compositional representation code including training and visualization for ENeRF-Outdoor dataset.
* `01/10/2023` We release [ENeRF-Outdoor](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md) dataset.

# ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video

> [Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://arxiv.org/abs/2112.01517)  
> Haotong Lin*, Sida Peng*, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao and Xiaowei Zhou \
> SIGGRAPH Asia 2022 conference track  
> [Project Page](https://zju3dv.github.io/enerf)

## Installation

### Set up the python environment

```
conda create -n enerf python=3.8
conda activate enerf
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html # Important!
pip install -r requirements.txt
```

### Set up datasets

#### 0. Set up workspace
The workspace is the disk directory that stores datasets, training logs, checkpoints and results. Please ensure it has enough space. 
```
export workspace=$PATH_TO_YOUR_WORKSPACE
```
   
#### 1. Pre-trained model

Download the pretrained model from [dtu_pretrain](https://zjueducn-my.sharepoint.com/:f:/g/personal/haotongl_zju_edu_cn/Elsj6QFXMVVBvjaqR2CFWrUBrRMfTjFiYpVcA5BAqqM3gA?e=vL8tiQ) (Pretrained on DTU dataset.)

Put it into `$workspace/trained_model/enerf/dtu_pretrain/latest.pth`.

#### 2. DTU
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet)
and unzip. [MVSNeRF](https://github.com/apchenstu/mvsnerf) provide a [DTU example](https://1drv.ms/u/s!AjyDwSVHuwr8zhAAXh7x5We9czKj?e=oStQ48), please follow with the example's folder structure.

```
mv dtu_example.zip $workspace
cd $workspace
unzip dtu_example.zip
```
This script only shows the example directory structure. You should download all the scenes in the DTU dataset and organize the data according to the example directory structure. Otherwise you can only do evaluation and fine-tuning on the example data.

#### 2. NeRF Synthetic and Real Forward-facing
Download the NeRF Synthetic and Real Forward-facing datasets from [NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and unzip them to $workspace. 
You should have the following directory.
```
$workspace/nerf_llff_data
$workspace/nerf_synthetic
```
#### 3. ZJU-MoCap

Download the ZJU-MoCap dataset from [NeuralBody](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset).
Put it into $workspace/zju_mocap/CoreView_313.

#### 4. ENeRF-Outdoor

Download the ENeRF-Outdoor dataset from this [link](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md).
Put it into $workspace/enerf_outdoor/actor1.
<!-- #### 5. DynamicCap -->
<!-- #### 6. Custom Data -->

## Training and fine-tuning

### Training
Use the following command to train a generalizable model on DTU.
```
python train_net.py --cfg_file configs/enerf/dtu_pretrain.yaml 
```

Our code also supports multi-gpu training. The published pretrained model was trained for 138000 iterations with 4 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=4 train_net.py --cfg_file configs/enerf/dtu_pretrain.yaml distributed True gpus 0,1,2,3
```


### Fine-tuning

```
cd $workspace/trained_model/enerf
mkdir dtu_ft_scan114
cp dtu_pretrain/138.pth dtu_ft_scan114
cd $codespace # codespace is the directory of the ENeRF code
python train_net.py --cfg_file configs/enerf/dtu/scan114.yaml
```

Fine-tuning for 3000 and 11000 iterations takes about 11 minutes and 40 minutes, respectively, on our test machine ( i9-12900K CPU, RTX 3090 GPU).

### Fine-tuning on the ZJU-MoCap dataset

```
python train_net.py --cfg_file configs/enerf/zjumocap/zjumocap_train.yaml
```

### Training on the ENeRF-Outdoor dataset (from scratch)

```
python train_net.py --cfg_file configs/enerf/enerf_outdoor/actor1.yaml
```

## Evaluation

### Evaluate the pretrained model on DTU

Use the following command to evaluate the pretrained model on DTU.
```
python run.py --type evaluate --cfg_file configs/enerf/dtu_pretrain.yaml enerf.cas_config.render_if False,True enerf.cas_config.volume_planes 48,8 enerf.eval_depth True
```


```
{'psnr': 27.60513418439332, 'ssim': 0.9570619, 'lpips': 0.08897018397692591}
{'abs': 4.2624497, 'acc_2': 0.8003020328362158, 'acc_10': 0.9279663826227568}
{'mvs_abs': 4.4139433, 'mvs_acc_2': 0.7711405202036934, 'mvs_acc_10': 0.9262374398033109}
FPS:  21.778975517304048
```

21.8 FPS@512x640 is tested on a desktop with an Intel i9-12900K CPU and an RTX 3090 GPU. **Add the "save_result True" parameter at the end of the command to save the rendering result.**

### Evaluate the pretrained model on LLFF and NeRF datasets

```
python run.py --type evaluate --cfg_file configs/enerf/nerf_eval.yaml
```

```
python run.py --type evaluate --cfg_file configs/enerf/llff_eval.yaml
```

### Evaluate the pretrained model on ZJU-MoCap dataset.

```
python run.py --type evaluate --cfg_file configs/enerf/zjumocap_eval.yaml
```

```
==============================
CoreView_313_level1 psnr: 31.48 ssim: 0.971 lpips:0.042
{'psnr': 31.477305846323087, 'ssim': 0.9714806, 'lpips': 0.04184799361974001}
==============================
FPS:  49.24468263992353
```

### Visualization for ENeRF-Outdoor dataset.

```
python run.py --type visualize --cfg_file configs/enerf/enerf_outdoor/actor1_path.yaml
```

## Interactive Rendering

We release the interactive rendering GUI for ZJU-MoCap dataset.

```
python gui_human.py --cfg_file configs/enerf/interactive/zjumocap.yaml
```

```
Usage:

Mouse wheel:          Zoom in/out
Mouse left button:    Move
Mouse right button:   Rotate
Keyboard a:           Align #  Hold down a and then use the mouse right button to rotate the object for a good rendering trajectory
Keyboard s:           Snap
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{lin2022enerf,
  title={Efficient Neural Radiance Fields for Interactive Free-viewpoint Video},
  author={Lin, Haotong and Peng, Sida and Xu, Zhen and Yan, Yunzhi and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2022}
}
```

