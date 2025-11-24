"""
Evaluate JAFAR performance on DAVIS video segmentation. Based on https://github.com/davisvideochallenge/davis2017-evaluation and https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py

Code adapted by: Saksham Suri and Matthew Walmer

Sample Usage:
python eval_davis.py
python eval_davis.py backbone.name=vit_small_patch14_reg4_dinov2 model=jafar
"""

import argparse
import copy
import datetime
import glob
import math
import os
import queue
import sys
import time
import warnings
from collections import defaultdict
from urllib.request import urlopen

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rich.console import Console
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training import load_multiple_backbones

torch.multiprocessing.set_sharing_strategy("file_system")


# ============================================================================
# DAVIS Evaluation Classes and Functions (Self-contained)
# ============================================================================


class DAVISDataset(object):
    """Simplified DAVIS dataset class for evaluation"""

    SUBSET_OPTIONS = ["train", "val", "test-dev", "test-challenge"]
    TASKS = ["semi-supervised", "unsupervised"]
    VOID_LABEL = 255

    def __init__(self, root, task="semi-supervised", subset="val", sequences="all", resolution="480p"):
        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, "JPEGImages", resolution)
        self.mask_path = os.path.join(self.root, "Annotations", resolution)
        self.imagesets_path = os.path.join(self.root, "ImageSets", "2017")

        if sequences == "all":
            with open(os.path.join(self.imagesets_path, f"{self.subset}.txt"), "r") as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]

        self.sequences = defaultdict(dict)
        for seq in sequences_names:
            images = np.sort(glob.glob(os.path.join(self.img_path, seq, "*.jpg"))).tolist()
            self.sequences[seq]["images"] = images
            masks = np.sort(glob.glob(os.path.join(self.mask_path, seq, "*.png"))).tolist()
            self.sequences[seq]["masks"] = masks

    def get_sequences(self):
        return list(self.sequences.keys())

    def get_all_masks(self, sequence, separate_objects_masks=False):
        """Get all ground truth masks for a sequence"""
        masks = []
        mask_ids = []

        for mask_path in self.sequences[sequence]["masks"]:
            mask = np.array(Image.open(mask_path))
            masks.append(mask)
            mask_ids.append(os.path.splitext(os.path.basename(mask_path))[0])

        masks = np.array(masks)

        if separate_objects_masks:
            # Convert to separate object masks
            num_objects = int(np.max(masks))
            if num_objects > 0:
                object_masks = np.zeros((num_objects, *masks.shape))
                for obj_id in range(1, num_objects + 1):
                    object_masks[obj_id - 1] = (masks == obj_id).astype(np.uint8)
                void_masks = np.zeros_like(masks[0])  # No void pixels for simplicity
                return object_masks, void_masks, mask_ids

        return masks, mask_ids


class DAVISResults(object):
    """Results reader for DAVIS evaluation"""

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _read_mask(self, sequence, frame_id):
        mask_path = os.path.join(self.root_dir, sequence, f"{frame_id}.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        return np.array(Image.open(mask_path))

    def read_masks(self, sequence, masks_id):
        """Read all prediction masks for a sequence"""
        mask_0 = self._read_mask(sequence, masks_id[0])
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)

        # Convert to separate object masks
        num_objects = int(np.max(masks))
        if num_objects == 0:
            return np.zeros((1, *masks.shape))

        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks


def davis_db_eval_iou(annotation, segmentation, void_pixels=None):
    """Compute region similarity as the Jaccard Index"""
    assert (
        annotation.shape == segmentation.shape
    ), f"Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match."

    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def davis_f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """Compute F-measure for boundary accuracy"""
    assert foreground_mask.shape == gt_mask.shape

    if void_pixels is not None:
        assert foreground_mask.shape == void_pixels.shape
        # Apply void mask
        foreground_mask = foreground_mask.copy()
        gt_mask = gt_mask.copy()
        foreground_mask[void_pixels] = 0
        gt_mask[void_pixels] = 0

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)

    from scipy.ndimage import distance_transform_edt

    # Distance transform
    fg_dist = distance_transform_edt(1 - fg_boundary)
    gt_dist = distance_transform_edt(1 - gt_boundary)

    # Precision: how many predicted boundary pixels are close to gt
    precision = np.sum(fg_boundary * (gt_dist <= bound_pix)) / (np.sum(fg_boundary) + 1e-10)

    # Recall: how many gt boundary pixels are close to predicted
    recall = np.sum(gt_boundary * (fg_dist <= bound_pix)) / (np.sum(gt_boundary) + 1e-10)

    # F-measure
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    return f_score


def _seg2bmap(seg, width=None, height=None):
    """Convert segmentation to boundary map"""
    seg = seg.astype(bool)
    seg = seg.astype(np.uint8)

    # Compute edges using Sobel operator
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    edges_x = cv2.filter2D(seg.astype(np.float32), -1, kernel_x)
    edges_y = cv2.filter2D(seg.astype(np.float32), -1, kernel_y)
    edges = np.sqrt(edges_x**2 + edges_y**2)

    # Threshold to get binary boundary
    bmap = edges > 0.1

    return bmap.astype(bool)


def davis_db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    """Compute boundary F-measure"""
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape

    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :]
            f_res[frame_id] = davis_f_measure(
                segmentation[frame_id, :, :], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th
            )
    elif annotation.ndim == 2:
        f_res = davis_f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f"Annotation and segmentation must be 2D or 3D arrays, got {annotation.ndim}D")

    return f_res


def davis_db_statistics(per_frame_values):
    """Compute mean, recall and decay from per-frame evaluation"""
    # Strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i] : ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


class DAVISEvaluation(object):
    """Complete DAVIS evaluation class"""

    def __init__(self, davis_root, task="semi-supervised", gt_set="val", sequences="all"):
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVISDataset(root=davis_root, task=task, subset=gt_set, sequences=sequences)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            print("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            return None, None
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)

        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if "J" in metric:
                j_metrics_res[ii, :] = davis_db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if "F" in metric:
                f_metrics_res[ii, :] = davis_db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    def evaluate(self, res_path, metric=("J", "F"), debug=False):
        """Run complete DAVIS evaluation"""
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if "T" in metric:
            raise ValueError("Temporal metric not supported!")
        if "J" not in metric and "F" not in metric:
            raise ValueError("Metric possible values are J for IoU or F for Boundary")

        # Containers
        metrics_res = {}
        if "J" in metric:
            metrics_res["J"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if "F" in metric:
            metrics_res["F"] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = DAVISResults(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            try:
                all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
                if self.task == "semi-supervised":
                    all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
                all_res_masks = results.read_masks(seq, all_masks_id)

                if self.task == "semi-supervised":
                    j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
                else:
                    # For unsupervised, use semi-supervised for now (simplified)
                    j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)

                if j_metrics_res is None:
                    continue

                for ii in range(all_gt_masks.shape[0]):
                    seq_name = f"{seq}_{ii+1}"
                    if "J" in metric:
                        JM, JR, JD = davis_db_statistics(j_metrics_res[ii])
                        metrics_res["J"]["M"].append(JM)
                        metrics_res["J"]["R"].append(JR)
                        metrics_res["J"]["D"].append(JD)
                        metrics_res["J"]["M_per_object"][seq_name] = JM
                    if "F" in metric:
                        FM, FR, FD = davis_db_statistics(f_metrics_res[ii])
                        metrics_res["F"]["M"].append(FM)
                        metrics_res["F"]["R"].append(FR)
                        metrics_res["F"]["D"].append(FD)
                        metrics_res["F"]["M_per_object"][seq_name] = FM

                # Show progress
                if debug:
                    print(f"Evaluated sequence: {seq}")

            except Exception as e:
                print(f"Error evaluating sequence {seq}: {e}")
                continue

        return metrics_res


@torch.no_grad()
def eval_video_tracking_davis(
    cfg, backbone, upsampler, transform, frame_list, video_dir, first_seg, seg_ori, color_palette, log_print
):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create output directory based on model configuration
    exp_name = f"davis_vidseg_{cfg.eval.ups_factor}_{cfg.model.name}_{cfg.backbone.name}"
    output_subdir = os.path.join(output_dir, exp_name)
    os.makedirs(output_subdir, exist_ok=True)

    video_folder = os.path.join(output_subdir, video_dir.split("/")[-1])
    os.makedirs(video_folder, exist_ok=True)

    # Setup device and normalization
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup normalization transforms for backbone
    mean_bck = backbone.config["mean"]
    std_bck = backbone.config["std"]
    mean_bck_tensor = torch.tensor(mean_bck, device=device).view(1, 3, 1, 1)
    std_bck_tensor = torch.tensor(std_bck, device=device).view(1, 3, 1, 1)

    # Upsampler normalization (ImageNet standard)
    mean_ups = (0.485, 0.456, 0.406)
    std_ups = (0.229, 0.224, 0.225)
    mean_ups_tensor = torch.tensor(mean_ups, device=device).view(1, 3, 1, 1)
    std_ups_tensor = torch.tensor(std_ups, device=device).view(1, 3, 1, 1)

    # The queue stores the n preceding frames
    que = queue.Queue(cfg.eval.n_last_frames)

    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0], backbone.config["ps"])

    # extract first frame feature using JAFAR pipeline
    frame1_feat = extract_feature(
        cfg,
        backbone,
        upsampler,
        transform,
        frame1,
        mean_bck_tensor,
        std_bck_tensor,
        mean_ups_tensor,
        std_ups_tensor,
    )
    frame1_feat = F.interpolate(frame1_feat, size=first_seg.shape[-2:], mode="bilinear", align_corners=False)
    frame1_feat = rearrange(frame1_feat, "1 c h w -> c (h w)")

    # saving first segmentation
    out_path = os.path.join(video_folder, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None

    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt], backbone.config["ps"])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]
        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(
            cfg,
            backbone,
            upsampler,
            transform,
            frame_tar,
            used_frame_feats,
            used_segs,
            mask_neighborhood,
            mean_bck_tensor,
            std_bck_tensor,
            mean_ups_tensor,
            std_ups_tensor,
        )

        # pop out oldest frame if necessary
        if que.qsize() == cfg.eval.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(
            frame_tar_avg,
            size=[v * backbone.config["ps"] // cfg.eval.ups_factor for v in frame_tar_avg.shape[-2:]],
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )[0]
        frame_tar_avg = frame_tar_avg.squeeze(0)
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = frame_list[cnt].split("/")[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


def restrict_neighborhood(cfg, h, w):
    """
    Ultra-fast approach using meshgrid and advanced indexing
    """
    size_mask = cfg.eval.size_mask_neighborhood

    # Create coordinate meshgrids
    query_i, query_j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    source_i, source_j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    # Flatten coordinates
    query_i_flat = query_i.flatten().unsqueeze(1)  # (h*w, 1)
    query_j_flat = query_j.flatten().unsqueeze(1)  # (h*w, 1)
    source_i_flat = source_i.flatten().unsqueeze(0)  # (1, h*w)
    source_j_flat = source_j.flatten().unsqueeze(0)  # (1, h*w)

    # Calculate distances using broadcasting
    i_dist = torch.abs(query_i_flat - source_i_flat)  # (h*w, h*w)
    j_dist = torch.abs(query_j_flat - source_j_flat)  # (h*w, h*w)

    # Create mask
    mask = ((i_dist <= size_mask) & (j_dist <= size_mask)).float()

    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if mask_cnt.max() > 0:
            mask_cnt = mask_cnt - mask_cnt.min()
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


def label_propagation(
    cfg,
    backbone,
    upsampler,
    transform,
    frame_tar,
    list_frame_feats,
    list_segs,
    mask_neighborhood,
    mean_bck_tensor,
    std_bck_tensor,
    mean_ups_tensor,
    std_ups_tensor,
):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape

    ## we only need to extract feature of the target frame
    feat_tar = extract_feature(
        cfg,
        backbone,
        upsampler,
        transform,
        frame_tar,
        mean_bck_tensor,
        std_bck_tensor,
        mean_ups_tensor,
        std_ups_tensor,
    )
    feat_tar = F.interpolate(feat_tar, size=(h, w), mode="bilinear", align_corners=False)
    feat_tar = rearrange(feat_tar, "1 c h w -> (h w) c")
    return_feat_tar = feat_tar.T  # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats)  # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)  # nmb_context x h*w (tar: query) x h*w (source: keys)

    if cfg.eval.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(cfg, h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w)  # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=cfg.eval.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T  # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood


@torch.no_grad()
def extract_feature(
    cfg,
    backbone,
    upsampler,
    transform,
    frame,
    mean_bck_tensor,
    std_bck_tensor,
    mean_ups_tensor,
    std_ups_tensor,
):
    """Extract one frame feature using JAFAR pipeline."""
    # Transform frame and move to device
    frame_tensor = transform(frame).unsqueeze(0).cuda()
    frame_tensor = F.interpolate(
        frame_tensor,
        size=[v // backbone.config["ps"] * backbone.config["ps"] for v in frame_tensor.shape[-2:]],
        mode="bilinear",
        align_corners=False,
    )

    # Prepare image for backbone (normalize)
    img_bck = (frame_tensor - mean_bck_tensor) / std_bck_tensor
    lr_feats = backbone(img_bck)

    # Upsample features
    img_ups = (frame_tensor - mean_ups_tensor) / std_ups_tensor

    # Resize input for upsampler to match target dimensions
    hr_size = [v * cfg.eval.ups_factor for v in lr_feats.shape[-2:]]
    img_ups = F.interpolate(img_ups, size=hr_size, mode="bicubic", align_corners=False)

    hr_feats = upsampler(img_ups, lr_feats, hr_size)
    return hr_feats


def imwrite_indexed(filename, array, color_palette):
    """Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format="PNG")


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if n_dims is None:
        n_dims = int(y_tensor.max() + 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir, "*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_frame(frame_dir, ps=None):
    img = Image.open(frame_dir)
    ori_w, ori_h = img.size
    return img, ori_h, ori_w


def read_seg(seg_dir, ps, factor):
    seg = Image.open(seg_dir)
    _w, _h = seg.size  # note PIL.Image.Image's size is (w, h)
    small_seg = np.array(seg.resize((_w // ps * factor, _h // ps * factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)


def run_video_segmentation(cfg, backbone, upsampler, log_print):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split("\n")[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)

    video_list = open(os.path.join(cfg.dataset.dataroot, "ImageSets/2017/val.txt")).readlines()

    start_time = time.time()

    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        log_print(f"[{i}/{len(video_list)}] Begin to segmentate video {video_name}.")

        video_dir = os.path.join(cfg.dataset.dataroot, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")

        # Read segmentation with appropriate scaling
        first_seg, seg_ori = read_seg(seg_path, backbone.config["ps"], cfg.eval.ups_factor)

        eval_video_tracking_davis(
            cfg, backbone, upsampler, transform, frame_list, video_dir, first_seg, seg_ori, color_palette, log_print
        )

    end_time = time.time()
    log_print(f"[bold blue]Total time taken: {(end_time - start_time) / 60:.2f} minutes[/bold blue]")
    log_print(f"[bold blue]Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB[/bold blue]")

    # Run comprehensive DAVIS evaluation
    log_print("[bold cyan]Running comprehensive DAVIS eval...[/bold cyan]")
    run_davis_evaluation(cfg, log_print)


def run_davis_evaluation(cfg, log_print):
    """Run comprehensive DAVIS evaluation using the integrated evaluation method"""
    output_dir = HydraConfig.get().runtime.output_dir
    exp_name = f"davis_vidseg_{cfg.eval.ups_factor}_{cfg.model.name}_{cfg.backbone.name}"
    results_path = os.path.join(output_dir, exp_name)

    if not os.path.exists(results_path):
        log_print(f"[bold red]Results path does not exist: {results_path}[/bold red]")
        return

    # Check if results already exist
    csv_name_global = "global_results-val.csv"
    csv_name_per_sequence = "per-sequence_results-val.csv"
    csv_name_global_path = os.path.join(results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(results_path, csv_name_per_sequence)

    if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
        log_print("[bold yellow]Using precomputed DAVIS results...[/bold yellow]")
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        log_print("[bold cyan]Computing DAVIS evaluation metrics...[/bold cyan]")

        # Create dataset and evaluate
        dataset_eval = DAVISEvaluation(davis_root=cfg.dataset.dataroot, task="semi-supervised", gt_set="val")
        metrics_res = dataset_eval.evaluate(results_path)

        if not metrics_res or "J" not in metrics_res or "F" not in metrics_res:
            log_print("[bold red]DAVIS evaluation failed![/bold red]")
            return

        J, F = metrics_res["J"], metrics_res["F"]

        # Generate dataframe for the general results
        g_measures = ["J&F-Mean", "J-Mean", "J-Recall", "J-Decay", "F-Mean", "F-Recall", "F-Decay"]
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.0
        g_res = np.array(
            [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]), np.mean(F["D"])]
        )
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)

        with open(csv_name_global_path, "w") as f:
            table_g.to_csv(f, index=False, float_format="%.3f")
        log_print(f"[bold green]Global results saved in {csv_name_global_path}[/bold green]")

        # Generate a dataframe for the per sequence results
        seq_names = list(J["M_per_object"].keys())
        seq_measures = ["Sequence", "J-Mean", "F-Mean"]
        J_per_object = [J["M_per_object"][x] for x in seq_names]
        F_per_object = [F["M_per_object"][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)

        with open(csv_name_per_sequence_path, "w") as f:
            table_seq.to_csv(f, index=False, float_format="%.3f")
        log_print(f"[bold green]Per-sequence results saved in {csv_name_per_sequence_path}[/bold green]")

    # Print the results
    log_print("[bold blue]--------------------------- Global DAVIS Results ---------------------------[/bold blue]")
    log_print(table_g.to_string(index=False))
    log_print("[bold blue]---------- Per-sequence DAVIS Results ----------[/bold blue]")
    log_print(table_seq.to_string(index=False))

    # Also save as JSON for compatibility
    metrics_dict = table_g.iloc[0].to_dict()
    import json

    with open(os.path.join(output_dir, "davis_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)
    log_print(f"[bold cyan]DAVIS metrics saved to: {os.path.join(output_dir, 'davis_metrics.json')}[/bold cyan]")


@hydra.main(config_path="../config", config_name="eval_video_seg", version_base=None)
def main(cfg: DictConfig):
    # Setup dual console logging (terminal and file)
    terminal_console = Console()
    current_run_dir = HydraConfig.get().runtime.output_dir
    log_file_path = os.path.join(current_run_dir, "eval_davis.log")
    file_console = Console(file=open(log_file_path, "w"))

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    # Start logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting DAVIS Video Segmentation Evaluation at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = "cuda"
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")

    # Load backbone and upsampler using Hydra configuration
    backbone_configs = [{"name": cfg.backbone.name}]
    backbones = load_multiple_backbones(cfg, backbone_configs, device="cuda")[0]
    backbone = backbones[0].eval().cuda()
    upsampler = instantiate(cfg.model).eval().cuda()

    # Load model checkpoint if provided
    if cfg.eval.model_ckpt:
        checkpoint = torch.load(cfg.eval.model_ckpt, map_location=device, weights_only=False)
        if cfg.model.name == "featup":
            new_ckpts = {
                k.replace("model.1.", "norm."): v
                for k, v in checkpoint["state_dict"].items()
                if "upsampler" in k or "model.1.norm" in k
            }
            upsampler.load_state_dict(new_ckpts, strict=True)
        else:
            upsampler.load_state_dict(checkpoint, strict=True)
        log_print(f"[green]Loaded model from checkpoint: {cfg.eval.model_ckpt}[/green]")
    else:
        log_print(f"[yellow]No model checkpoint provided, using untrained model[/yellow]")

    run_video_segmentation(cfg, backbone, upsampler, log_print)

    # Log completion before closing file
    log_print(f"[bold blue]Evaluation completed. Log saved to: {log_file_path}[/bold blue]")

    # Close file console
    file_console.file.close()


if __name__ == "__main__":
    main()
