# Reproducibility 

⚠️ Before launching commands, please make sure to set the `dataroot` variable in [config/base.yaml](config/base.yaml) to the path where the datasets are stored.

General instructions:
- Change the VFM: add `model.backbone=<backbone_name>` to the command line (e.g. `model.backbone=franca_vitb16`).
- Change the dataset: add `dataset=<dataset_name>` to the command line (e.g. `dataset=cityscapes`).

For debugging, add `HYDRA_FULL_ERROR=1` before the command to get full stack traces.

# 🔄 Training
Training NAF take less than 2 hours consuming less than 8GB of GPU memory on a single NVIDIA A100.

To train the VFM-agnostic upsampler using NAF, simply run:
``` python
python train.py hydra.run.dir=output/naf
```

To train the denoising model using NAF, simply run:
``` python
python denoising.py denoising.noise_type=gaussian
```
You can select salt-and-pepper noise by setting `denoising.noise_type=salt_pepper`.