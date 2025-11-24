# Datasets

Create a data folder corresponding to the root data directory represented in the configuration file by `data_root: data/`.

We train NAF using the ImageNet dataset, but it can be trained on any dataset of natural images. Then we evaluate NAF on several downstream tasks using the following datasets:
- Semantic Segmentation: Cityscapes, PASCAL VOC 2012, ADE20K, KITTI-360
- Depth Estimation: NYU Depth V2
- Video Object Segmentation: DAVIS 2017

To download the datasets used for training and evaluation, please refer to the [docs/data.sh](data.sh) file or see the official dataset documentation.

The data folder should have the following structure:

```
data/
(training)
└── ImageNet/ 
(evaluation)
├── cityscapes/ 
├── VOCdevkit/
├── ADEChallengeData2016/ 
├── KITTI-360/ 
├── DAVIS/
└── ...
```