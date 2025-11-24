import json
import os
import random
from collections import namedtuple
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# https://github.com/autonomousvision/kitti360Scripts/blob/32f6d64eef27c32b52c542b4e95be8af9c9c6444/kitti360scripts/helpers/labels.py
# a label and all meta information
Label = namedtuple(
    "Label",
    ["name", "id", "kittiId", "trainId", "category", "categoryId", "hasInstances", "ignoreInEval", "ignoreInInst", "color"],
)


labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label("unlabeled", 0, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("ego vehicle", 1, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("rectification border", 2, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("out of roi", 3, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("static", 4, -1, 255, "void", 0, False, True, True, (0, 0, 0)),
    Label("dynamic", 5, -1, 255, "void", 0, False, True, True, (111, 74, 0)),
    Label("ground", 6, -1, 255, "void", 0, False, True, True, (81, 0, 81)),
    Label("road", 7, 1, 0, "flat", 1, False, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 3, 1, "flat", 1, False, False, False, (244, 35, 232)),
    Label("parking", 9, 2, 255, "flat", 1, False, True, True, (250, 170, 160)),
    Label("rail track", 10, 10, 255, "flat", 1, False, True, True, (230, 150, 140)),
    Label("building", 11, 11, 2, "construction", 2, True, False, False, (70, 70, 70)),
    Label("wall", 12, 7, 3, "construction", 2, False, False, False, (102, 102, 156)),
    Label("fence", 13, 8, 4, "construction", 2, False, False, False, (190, 153, 153)),
    Label("guard rail", 14, 30, 255, "construction", 2, False, True, True, (180, 165, 180)),
    Label("bridge", 15, 31, 255, "construction", 2, False, True, True, (150, 100, 100)),
    Label("tunnel", 16, 32, 255, "construction", 2, False, True, True, (150, 120, 90)),
    Label("pole", 17, 21, 5, "object", 3, True, False, True, (153, 153, 153)),
    Label("polegroup", 18, -1, 255, "object", 3, False, True, True, (153, 153, 153)),
    Label("traffic light", 19, 23, 6, "object", 3, True, False, True, (250, 170, 30)),
    Label("traffic sign", 20, 24, 7, "object", 3, True, False, True, (220, 220, 0)),
    Label("vegetation", 21, 5, 8, "nature", 4, False, False, False, (107, 142, 35)),
    Label("terrain", 22, 4, 9, "nature", 4, False, False, False, (152, 251, 152)),
    Label("sky", 23, 9, 10, "sky", 5, False, False, False, (70, 130, 180)),
    Label("person", 24, 19, 11, "human", 6, True, False, False, (220, 20, 60)),
    Label("rider", 25, 20, 12, "human", 6, True, False, False, (255, 0, 0)),
    Label("car", 26, 13, 13, "vehicle", 7, True, False, False, (0, 0, 142)),
    Label("truck", 27, 14, 14, "vehicle", 7, True, False, False, (0, 0, 70)),
    Label("bus", 28, 34, 15, "vehicle", 7, True, False, False, (0, 60, 100)),
    Label("caravan", 29, 16, 255, "vehicle", 7, True, True, True, (0, 0, 90)),
    Label("trailer", 30, 15, 255, "vehicle", 7, True, True, True, (0, 0, 110)),
    Label("train", 31, 33, 16, "vehicle", 7, True, False, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, 17, "vehicle", 7, True, False, False, (0, 0, 230)),
    Label("bicycle", 33, 18, 18, "vehicle", 7, True, False, False, (119, 11, 32)),
    Label("garage", 34, 12, 2, "construction", 2, True, True, True, (64, 128, 128)),
    Label("gate", 35, 6, 4, "construction", 2, False, True, True, (190, 153, 153)),
    Label("stop", 36, 29, 255, "construction", 2, True, True, True, (150, 120, 90)),
    Label("smallpole", 37, 22, 5, "object", 3, True, True, True, (153, 153, 153)),
    Label("lamp", 38, 25, 255, "object", 3, True, True, True, (0, 64, 64)),
    Label("trash bin", 39, 26, 255, "object", 3, True, True, True, (0, 128, 192)),
    Label("vending machine", 40, 27, 255, "object", 3, True, True, True, (128, 64, 0)),
    Label("box", 41, 28, 255, "object", 3, True, True, True, (64, 64, 128)),
    Label("unknown construction", 42, 35, 255, "void", 0, False, True, True, (102, 0, 0)),
    Label("unknown vehicle", 43, 36, 255, "void", 0, False, True, True, (51, 0, 51)),
    Label("unknown object", 44, 37, 255, "void", 0, False, True, True, (32, 32, 32)),
    Label("license plate", -1, -1, -1, "vehicle", 7, False, True, True, (0, 0, 142)),
]


class KITTIDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        include_labels: bool = True,
        num_classes: int = 19,
        tag: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        self.num_classes = num_classes
        self.tag = tag

        # Create mapping from id to trainId
        self.id_to_trainid = {label.id: label.trainId for label in labels}

        # Initialize paths
        self.image_paths = []
        self.label_paths = []

        # Set up split management
        self.split_file = os.path.join(os.path.dirname(__file__), "kitti360", f"{split}_split.json")

        # Create split if needed
        if not os.path.exists(self.split_file):
            self._create_split()

        # Load paths from split file
        self._load_split()

    def _create_split(self):
        """Create and save train/val split if it doesn't exist"""
        # Collect all valid paths first
        all_image_paths = []
        all_label_paths = []

        raw_base = os.path.join(self.root, "data_2d_raw")
        sem_base = os.path.join(self.root, "data_2d_semantics", "train")

        for drive in os.listdir(raw_base):
            drive_path = os.path.join(raw_base, drive)

            for camera in ["image_00"]:
                img_dir = os.path.join(drive_path, camera, "data_rect")
                lbl_dir = os.path.join(sem_base, drive, camera, "semantic")

                if os.path.exists(img_dir) and os.path.exists(lbl_dir):
                    for fname in os.listdir(img_dir):
                        if fname.endswith(".png"):
                            base_name = os.path.splitext(fname)[0]
                            img_path = os.path.join(img_dir, fname)
                            lbl_path = os.path.join(lbl_dir, f"{base_name}.png")

                            if os.path.exists(lbl_path):
                                all_image_paths.append(img_path)
                                all_label_paths.append(lbl_path)

        # Create 80-20 split
        random.seed(42)  # For reproducibility
        combined = list(zip(all_image_paths, all_label_paths))
        random.shuffle(combined)
        split_idx = int(0.8 * len(combined))

        train_split = combined[:split_idx]
        val_split = combined[split_idx:]

        # Create directory if needed
        split_dir = os.path.dirname(self.split_file)
        os.makedirs(split_dir, exist_ok=True)

        # Save splits
        with open(os.path.join(split_dir, "train_split.json"), "w") as f:
            json.dump(train_split, f)

        with open(os.path.join(split_dir, "val_split.json"), "w") as f:
            json.dump(val_split, f)

    def _load_split(self):
        """Load paths from existing split file"""
        with open(self.split_file, "r") as f:
            split_data = json.load(f)

        self.image_paths, self.label_paths = zip(*split_data)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        batch = {}
        seed = random.randint(0, 2**32 - 1)

        # Load image
        random.seed(seed)
        torch.manual_seed(seed)
        img = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        batch["image"] = img
        batch["img_path"] = self.image_paths[index]

        if self.include_labels:
            # Load and process label
            random.seed(seed)
            torch.manual_seed(seed)
            label = Image.open(self.label_paths[index])

            if self.target_transform:
                label = self.target_transform(label)

            # Convert from id to trainId
            label_array = np.array(label)
            for id_val, trainid_val in self.id_to_trainid.items():
                label_array[label_array == id_val] = trainid_val

            label = torch.from_numpy(label_array).to(torch.uint8)
            batch["label"] = label.squeeze(0)

        return batch
