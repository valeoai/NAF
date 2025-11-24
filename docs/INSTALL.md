# Installation Instructions
## Install Natten
NAF is based on natten. Natten is build for specific torch versions. You simply need to install the compatible natten version for your torch version. For instance:

```bash
# Version used for developping.
pip install natten==0.17.4+torch240cu118 -f https://shi-labs.com/natten/wheels/
```
For newer versions of torch or natten check this [url](https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md).
For older versions of torch or natten check this [url](https://shi-labs.com/natten/wheels/).

We have verified the following compatibilities:
```base
# For torch 2.7.0 + cu118
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install natten==0.20.1+torch270cu128 -f https://whl.natten.org

# For torch 2.4.0 + cu118
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install natten==0.17.3+torch200cu117 -f https://shi-labs.com/natten/wheels/
```

Do not hesitate to open an issue if you face any problem installing natten.

## Complete environment
We used the following environment for developping NAF. We recommend using [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to create the environment and using [uv](https://github.com/astral-sh/uv) to speed up pip installs.

``` bash
micromamba create -n naf python==3.10.14  -y -c defaults
micromamba activate naf
pip install uv

# Install torch and natten
uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118 
pip install natten==0.17.4+torch240cu118 -f https://shi-labs.com/natten/wheels/

# Install other dependencies
uv pip install einops==0.8.0 numpy==1.24.4 timm==1.0.22 plotly==6.0.0 tensorboard==2.20.0 hydra-core==1.3.2 plotly==6.0.0 matplotlib==3.7.0 rich==14.2.0 torchmetrics==1.6.2 scipy==1.15.2 kornia==0.8.2 ipykernel ipympl pytest
```