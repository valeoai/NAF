import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DAVIS(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None, tag=None):
        """
        Args:
            root (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        videos = self.load_split(root, split)

        frames = []
        for video in videos:
            video_path = os.path.join(root, f"JPEGImages/480p/{video}/*/*.jpg")
            video_frames = sorted(glob.glob(video_path))
            frames += video_frames
        self.frames = frames
        self.transform = transform
        self.target_transform = target_transform
        self.tag = tag

    def load_split(self, root, split):
        """
        Load the split files for DAVIS dataset.
        Args:
            root (string): Directory with all the videos.
            split (string): Split name (train, val, test).
        Returns:
            list: List of video paths.
        """
        split_path = os.path.join(root, "ImageSets/2017/", f"{split}.txt")
        with open(split_path, "r") as f:
            videos = [line.strip() for line in f.readlines()]
        return videos

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        image_path = self.frames[index]
        annotation_path = image_path.replace("JPEGImages", "Annotations")
        batch = {}

        # Load
        image = Image.open(image_path).convert("RGB")
        target = Image.open(annotation_path)

        # Augment
        image = self.transform(image).squeeze(0)
        target = self.target_transform(target)
        target = target.squeeze(0)

        batch = {"image": image, "label": target}
        return batch
