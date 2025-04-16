# <p align="center"> PicoPose: Progressive Pixel-to-Pixel Correspondence Learning for Novel Object Pose Estimation </p>
####  <p align="center"> [Lihua Liu](https://github.com/foollh), [Jiehong Lin](https://jiehonglin.github.io/), [Zhenxin Liu](https://github.com/Liu-Zhen-Xin), [Kui Jia](http://kuijia.site/)</p>


<div align="center">
  <a href="https://arxiv.org/abs/2504.02617">
    <img src="https://img.shields.io/badge/arXiv-2504.02617-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</div>

![overview](pics/overview.jpg) 

PicoPose is a novel three-stage framework for zero-shot 6D pose estimation of unseen objects from RGB images. It first matches RGB features with rendered templates for coarse correspondences, then smooths the correspondences by globally regressing a 2D affine transformation, and finally learns correspondence offsets within local regions to achieve fine-grained correspondences. 


## Environment
```
conda create -n picopose python=3.9
conda activate picopose
python -m pip install -r requirements.txt
```


## Data Preparation

<details><summary>Click to expand</summary>

### Data Structure
Our data structure in [data](data) folder is constructed as follows:
```
data
├── MegaPose-Training-Data
    ├── MegaPose-GSO
        ├──google_scanned_objects
        └──train_pbr_web
    ├── MegaPose-ShapeNetCore
        ├──shapenetcorev2
        └──train_pbr_web
    ├── MegaPose-Templates  
        ├──GSO
        └──ShapeNetCore
├── BOP_Datasets   # https://bop.felk.cvut.cz/datasets/
    ├──tudl
    ├──lmo
    ├──ycbv
    ├──icbin
    ├──hb
    ├──itodd
    ├──tless
    └──templates
└── bop23_default_detections_for_task4
    └──cnos-fastsam
```

### Data Download
The training datasets you can download the rendered images of [c](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_challenge_2023_training_datasets.md) provided by BOP official in the respective `MegaPose-Training-Data/MegaPose-GSO/train_pbr_web` and `MegaPose-Training-Data/MegaPose-ShapeNetCore/train_pbr_web` folders. 

The [pre-processed object models](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/) of the two datasets provided by [MegePose](https://github.com/megapose6d/megapose6d) can be downloaded to the `MegaPose-Training-Data/MegaPose-GSO/google_scanned_objects` and `MegaPose-Training-Data/MegaPose-ShapeNetCore/shapenetcorev2` folders respectively.

To evaluate on BOP datasets, you can download the test data and object CAD models for the seven core datasets from the official [BOP](https://bop.felk.cvut.cz/datasets/) website.
We use the [CNOS detection results](https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip) on seven datasets officially provided by BOP.

</details>


## Rendering Templates
Before starting training and testing, please render the corresponding templates in advance. We mainly render the templates based on the code of [GigaPose]((https://github.com/nv-nguyen/gigapose)), and the entire rendering environment is on the docker provided by [MegaPose](https://github.com/megapose6d/megapose6d).
```
# Installing extra environments on megapose docker
bash rendering/inv_env.sh

# Rendering shapenetv2
python rendering/scripts/render_shapenet_templates.py 

# Rendering gso 
python rendering/scripts/render_gso_templates.py 

# Rendering bop
python rendering/scripts/render_bop_templates.py 
```


## Training on MegaPose Dataset
To train PicoPose, please prepare the training data and run the folowing command:
```
python run_train.py --model picopose --config config/base.yaml --version_id 0 
```


## Testing on BOP Dataset

To evaluate the model on BOP datasets, please run the following command:
```
python run_test.py --gpus 0 --model picopose --config config/base.yaml --dataset $DATASET --version_id 0 
```
The string "DATASET" could be set as `lmo`, `icbin`, `itodd`, `hb`, `tless`, `tudl` or `ycbv`. We also offer downloadable rendered templates [[link](https://drive.google.com/drive/u/0/folders/1EwzDEbZQrMhhsyTieLfy8rohqIK4Y5mj)]. 

One could also directly specify the checkpoint path for evaluation:
```
python run_test.py --gpus 0 --model picopose --config config/base.yaml --dataset $DATASET --version_id 0 --checkpoint_path $CHECKPOINT_PATH
```
The string "CHECKPOINT_PATH" is the path to the checkpoint saved during training. Our trained model is provided [here](https://drive.google.com/file/d/1hDDr0o4pEEHKi4QOUQ4zU0I-Ts5H2bI1/view?usp=sharing).


## Citation
If you find our work useful in your research, please consider citing:

    @article{liu2025picopose,
    title={PicoPose: Progressive Pixel-to-Pixel Correspondence Learning for Novel Object Pose Estimation},
    author={Liu, Lihua and Lin, Jiehong and Liu, Zhenxin and Jia, Kui},
    journal={arXiv preprint arXiv:2504.02617},
    year={2025}
    }


## Acknowledgements
- [MegaPose](https://github.com/megapose6d/megapose6d)
- [GigaPose](https://github.com/nv-nguyen/gigapose)
- [SAM-6D](https://github.com/JiehongLin/SAM-6D)
- [SCFlow](https://github.com/YangHai-1218/SCFlow)
