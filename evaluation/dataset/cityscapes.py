import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes


class CityscapesDataset(Dataset):
    BASE_DIR = "cityscapes"
    NUM_CLASS = 19

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        include_labels=True,
        num_classes=19,
        tag=None,
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

        # fmt: off
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype("int32")
        # fmt: on

        # Get image/mask path pairs
        self.cityscapes_dataset = Cityscapes(
            root=root,
            split=split,
            mode="fine",
            target_type=["instance", "semantic"],
            transform=None,
            target_transform=None,
        )
        # fmt: on

        if split == "train":
            assert self.__len__() == 2975
        elif split == "val":
            assert self.__len__() == 500

    def get_class_names(
        self,
    ):
        return [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

    def __len__(self):
        return len(self.cityscapes_dataset)

    def __getitem__(self, index):
        batch = {}

        # Load image
        data = self.cityscapes_dataset[index]
        img = data[0]
        mask = data[1][1]

        if self.transform:
            img = self.transform(img)
        batch["image"] = img

        if self.include_labels:
            if self.target_transform:
                mask = self.target_transform(mask)
            # Convert to tensor and match COCO/ADE20K format
            mask = torch.from_numpy(self._class_to_index(np.array(mask))).to(torch.uint8)
            batch["label"] = mask.squeeze()

        return batch

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert value in self._mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape).astype(np.int64)
