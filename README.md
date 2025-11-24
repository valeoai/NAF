# Official Implementation of *NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering.*

> [**NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering.**]<br>
> [Loick Chambon](https://loickch.github.io/),  [Paul Couairon](https://pcouairon.github.io/), [Eloi Zablocki](https://scholar.google.fr/citations?user=dOkbUmEAAAAJ&hl=fr), [Alexandre Boulch](https://boulch.eu/), [Nicolas Thome](https://thome.isir.upmc.fr/), [Matthieu Cord](https://cord.isir.upmc.fr/).<br> Valeo.ai, Sorbonne University, CNRS.<br> 


<table>
  <tr>
    <td align="center" width="50%">
      <img src='./asset/teasing.gif' width="90%">
    </td>
  </tr>
</table>

## 🎯 TL;DR

**Three simple steps:**
1. **Select any Vision Foundation Model** ([DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2](https://github.com/facebookresearch/dinov2), [RADIO](https://github.com/NVlabs/RADIO), [FRANCA](https://github.com/valeoai/Franca), [PE-CORE](https://github.com/facebookresearch/perception_models), [CLIP](https://github.com/openai/CLIP), [SAM](https://github.com/facebookresearch/segment-anything), etc.)
2. **Choose your target resolution** (up to 2K)
3. **Upsample features with NAF** — zero-shot, no retraining needed

**Why it works:** NAF combines classical filtering theory with modern attention mechanisms, learning adaptive kernels through Fourier space transformations.

## ⚡ News & Updates
- [ ] Release trained checkpoints for **NAF++**.
- [x] **[2025-11-25]** NAF has been uploaded on arXiv.
- [x] **[2025-11-24]** NAF code has been publicly released.

# 📜 Abstract

Vision Foundation Models produce **downsampled spatial features**, which are challenging for pixel-level tasks.  

❌ Traditional upsampling methods:  
* **Classical filters** – fast, generic, but fixed  (bilinear, bicubic, joint bilateral, guided)
* **Learnable VFM-specific upsamplers** – accurate, but need retraining (FeatUp, LiFT, JAFAR, LoftUp)

✅ **NAF (Neighborhood Attention Filtering)**:  
* Learns **adaptive spatial-and-content weights** using Cross-Scale Neighborhood Attention + RoPE  
* Works **zero-shot** for any VFM  
* Outperforms existing upsamplers on **multiple downstream tasks**  
* Efficient: scales up to 2K features, ~18 FPS for intermediate resolutions  
* Also effective for **image restoration**  


# 🥇 Results

NAF achieves **state-of-the-art performance** on:  

- Semantic Segmentation  
- Depth Estimation  
- Open-Vocabulary Tasks  
- Video Segmentation Propagation  

Tested with **multiple VFMs** (T-S-B-L-G-7B) and **datasets** (COCO, VOC, ADE20K, Cityscapes, KITTI-360, NYUv2)


## ✨ Downstream Tasks

<table>
  <tr>
  <td align="center">
    <img src="asset/results.png" style="width: 90%;">
  </td>
  </tr>
  
  <tr>
  <td align="center">
    <em>NAF achieves state-of-the-art performance on various benchmarks: Semantic Segmentation, Depth Estimation, Video Label Propagation, Open-Vocabulary beating previous VFM-specific and VFM-agnostic methods while being more efficient.</em>
  </td>
  </tr>
</table>

## 🔢 Filtering and Fourier Analysis

We found that NAF learns the Inverse Discrete Fourier Transform (IDFT) of the upsampling aggregation kernel, providing insights into the mechanism behind its results:

**NAF Filtering equation:**

$$
\mathbf{F}^{\mathrm{HR}}_{p} = \frac{1}{Z(p)} \sum_{q \in \mathcal{N}(p)} \exp\left(\frac{\langle Q_p, K_q \rangle}{\sqrt{d}} \right) \mathbf{F}^\mathrm{LR}_{q}
$$

**NAF Fourier interpretation:**

$$
S(x) \propto \sum_{c} \underbrace{r_p^{(c)} r_{q'}^{(c)}}_{\text{Amplitude}} \cos\!\left( \underbrace{\Psi_c}_{\text{Content Phase}} + \underbrace{\omega_c \Delta x}_{\text{Spatial Phase}} \right)
$$

📄 **More details in the paper!**

# 🔨 Setup

See the docs folder for detailed setup instructions concerning [installs](docs/INSTALL.md), [datasets](docs/DATASETS.md), [training](docs/TRAINING.md) and [evaluation](docs/EVALUATIONS.md).

Note that: Training NAF takes less than 2 hours consuming less than 8GB of GPU memory on a single NVIDIA A100.

# 📁 Repository Structure

The repository contains the important folders:
```
|-- configs/                # Configuration files
|-- docs/                   # Documentation
|-- evaluation/             # Scripts to reproduce results and datasets initialization
|-- notebooks/              # Jupyter notebooks for inference (of any VFM at any scale) and attention map visualizations
|-- src/                    # Source code of the project
|-- test/                   # Unit tests to compare model efficiency
```

## 🔄 Notebooks 
- Inference: [notebooks/inference.ipynb](notebooks/inference.ipynb) runs NAF upsampler on any VFM.

<table align="center">
  <tr>
    <td align="center">
      <img src="asset/inference.png" style="width: 70%;">
      <br>
      <em>NAF enables zero-shot feature upsampling across any Vision Foundation Model</em>
    </td>
  </tr>
</table>
<table align="center">
  <tr>
    <td align="center">
      <img src="asset/resolution.png" style="width: 90%;">
      <br>
      <em>Seamless upsampling from low-resolution to high-resolution features</em>
    </td>
  </tr>
</table>

- Attention Maps: [notebooks/attention_maps.ipynb](notebooks/attention_maps.ipynb) visualizes NAF neighborhood attention maps.

<table>
  <tr>
    <td align="center" width="50%">
      <img src='./asset/attention.gif' width="90%">
    </td>
  </tr>
  
  <tr>
    <td colspan="2" align="center">
      <em>Given a query point and a kernel size, we compute and show its neighborhood attention map.</em>
    </td>
  </tr>
</table>

## 🔍 Tests

We provide unit tests to evaluate the efficiency of different model configurations, including forward/backward runtime, GPU memory usage, GFLOPs, and number of parameters.  
All tests are available in the `test/` directory and we provide the test_results.json file with pre-computed results for reference computed on a 1 A100 40GB GPU.

## 👍 Acknowledgements

Many thanks to these excellent open source projects:
* https://github.com/SHI-Labs/NATTEN
* https://github.com/PaulCouairon/JAFAR
* https://github.com/mhamilton723/FeatUp
* https://github.com/saksham-s/lift/tree/main
* https://github.com/mc-lan/ProxyCLIP

To structure our code we used:
* [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)
* [Hydra](https://github.com/facebookresearch/hydra)
* [Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning)

Do not hesitate to look and support our previous feature upsampling work:
* https://github.com/PaulCouairon/JAFAR


## ✏️ Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry and putting a star on this repository. Feel free to open an issue for any questions.

```
@misc{chambon2025naf,
      title={NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering.}, 
      author={Loick Chambon and Paul Couairon and Eloi Zablocki and Alexandre Boulch and Nicolas Thome and Matthieu Cord},
      year={2025},
}
```