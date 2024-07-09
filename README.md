# MFDFA

## Introduction
 this paper proposes MFAFD: a cascaded model for

few-shot learning featuring parameter free attention mechanisms and a finite dis

crete space. Initially, the model employs a parameter free attention module in

the pretraining phase to facilitate cross-modal interactions, enhancing alignment

between spatial features of images and generated text prior to extracting global

features from images via CLIP. This bidirectional update of textual and visual

information addresses the issue of feature alignment. During training, the model

leverages a representation based on Finite Discrete Space (FDS), constructing a

finite discrete space foundation for textual and image features, effectively bridging

modal differences. Ultimately, using text as a baseline, the model predicts image

classification based on similarity weights between images and text. Through quan

titative and qualitative analyses, this study demonstrates that parameter free

attention mechanisms and finite discrete space modules significantly enhance the

performance of cascaded multimodal aggregation models. The model exhibits

robust performance in few-shot classification across multiple datasets. 

### Installation
Create a conda environment and install dependencies:
```bash
conda create -n MFDFA python=3.7
conda activate MFDFA

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Please download official ImageNet and other 10 datasets.

### Foundation Models
* The pre-tained weights of **CLIP** will be automatically downloaded by running.
* The prompts produced by **GPT-3** have been stored at `gpt_file/`.
* Please download **DINO's** pre-trained ResNet-50 from [here](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth), and put it under `dino/`.
* Please download **DALL-E's** generated images from [here](https://drive.google.com/drive/folders/1e249OgUFCmpfEDPsxCVR-nNb6Q1VaZVW?usp=sharing), and organize them with the official datasets like
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– ...
|–– dalle_imagenet/
|–– dalle_caltech-101/
|–– dalle_oxford_pets/
|–– ...
|–– sd_caltech-101/
```
* For Caltech-101 dataset, we also provide **Stable Diffusion's** images from [here](https://drive.google.com/drive/folders/1e249OgUFCmpfEDPsxCVR-nNb6Q1VaZVW?usp=sharing), and **ChatGPT's** prompts in `gpt_file/`.

## Get Started
### Configs
The running configurations for different `[dataset]` with `[k]` shots can be modified in `configs/[dataset]/[k]shot.yaml`, including visual encoders and hyperparamters. We have provided the configurations for reproducing the results in the paper. You can edit the `search_scale`, `search_step`, `init_beta` and `init_alpha` for fine-grained tuning and better results.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.

### Running
For 16-shot ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet/16shot.yaml
```



## Acknowledgement
This work benefits from [CLIP](https://github.com/openai/CLIP), [DINO](https://github.com/facebookresearch/dino), [DALL-E](https://github.com/borisdayma/dalle-mini) , [CuPL](https://github.com/sarahpratt/CuPL) and [CAFO](https://github.com/ZrrSkywalker/CaFo.). Thanks for their wonderful works.



## Contributors
Lixia Xue , Jiang Dong , Ronggui Wang , Juan Yang

