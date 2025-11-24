import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        include_labels: bool = True,
        num_classes: int = 21,
        tag: Optional[str] = None,
        year: str = "2012",
        download: bool = False,
        **kwargs
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        self.num_classes = num_classes
        self.tag = tag

        # Initialize torchvision's VOC dataset
        self.voc_dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set="train" if split == "train" else "val",
            download=download,
            transform=None,
            target_transform=None,
        )

        if split == "train":
            assert self.__len__() == 1464
        elif split == "val":
            assert self.__len__() == 1449

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, index):
        batch = {}
        seed = random.randint(0, 2**32 - 1)

        # Load image
        img_path, mask_path = (
            self.voc_dataset.images[index],
            self.voc_dataset.masks[index],
        )

        random.seed(seed)
        torch.manual_seed(seed)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        batch["image"] = img
        batch["img_path"] = img_path

        if self.include_labels:
            # Load and process mask
            random.seed(seed)
            torch.manual_seed(seed)
            mask = Image.open(mask_path)

            if self.target_transform:
                mask = self.target_transform(mask)

            # Convert to tensor and match COCO/ADE20K format
            mask = torch.from_numpy(np.array(mask)).to(torch.uint8)
            batch["label"] = mask.squeeze()

        return batch

    @property
    def pred_offset(self):
        return 0

    def get_class_names(self):
        """Return the class names for the VOC dataset."""
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
