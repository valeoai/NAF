import os
import sys

import torch
import torchvision.transforms as T
from torchvision.datasets import folder
from torchvision.datasets.vision import VisionDataset


def remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix) :]
    return s


class ImageDataset(VisionDataset):
    """
    modified from: https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    uses cached directory listing if available rather than walking directory
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        root_cache,
        loader=folder.default_loader,
        extensions=folder.IMG_EXTENSIONS,
        transform=None,
        is_valid_file=None,
        include_labels=False,
        **kwargs,
    ):
        super(ImageDataset, self).__init__(
            root,
            transform=transform,
        )
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        if root_cache is None:
            root_cache = root
        cache = root_cache.rstrip("/") + ".txt"
        print(cache)
        # cache = './val_paths.txt'
        if os.path.isfile(cache):
            print("Using directory list at: %s" % cache)
            with open(cache) as f:
                samples = []
                for line in f:
                    (path, idx) = line.strip().split(";")
                    samples.append((os.path.join(path), int(idx)))
        else:
            print("Walking directory: %s" % self.root)
            samples = folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            with open(cache, "w") as f:
                for line in samples:
                    path, label = line
                    f.write("%s;%d\n" % (remove_prefix(path, self.root).lstrip("/"), label))

        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.root + "\n" "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.include_labels = include_labels

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        batch = {
            "index": index,
            "image": sample,
            "target": target,
            "path": path,
        }

        if self.include_labels:
            target = self.targets[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            batch["label"] = target

        return batch

    def __len__(self):
        return len(self.samples)
