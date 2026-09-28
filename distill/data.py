"""Dataset construction and the class-subset plumbing."""

import logging

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .utils import worker_init_fn

log = logging.getLogger(__name__)

DATASETS = {"cifar10": (datasets.CIFAR10, 10), "cifar100": (datasets.CIFAR100, 100)}


def build_transforms(augment=True):
    """The eval transform, and the train transform with optional augmentation."""
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def build_datasets(name, root="./data", augment=True, download=True):
    """Return (train, train_eval, test, num_classes).

    `train_eval` is the training split under the eval transform, which is what
    the exemplar memory herds over: augmented features would make the class
    means noisy.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    cls, num_classes = DATASETS[name]
    train_tf, eval_tf = build_transforms(augment)

    try:
        train_set = cls(root=root, train=True, download=download, transform=train_tf)
        train_eval = cls(root=root, train=True, download=False, transform=eval_tf)
        test_set = cls(root=root, train=False, download=download, transform=eval_tf)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not load {name} from {root!r}: {exc}. "
            "Pass --root to point at the data, or drop --no-download to fetch it."
        ) from exc

    log.info("%s: %d train / %d test images, %d classes",
             name, len(train_set), len(test_set), num_classes)
    return train_set, train_eval, test_set, num_classes


class RemappedSubset(Dataset):
    """A subset of `base` whose labels are rewritten through `label_map`.

    Incremental tasks need the classes numbered in the order they arrive, not
    in the dataset's own order, so that the growing head stays contiguous.
    """

    def __init__(self, base, indices, label_map):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.label_map = label_map
        targets = getattr(base, "targets", None)
        if targets is not None:
            missing = {int(targets[i]) for i in self.indices} - set(label_map)
            if missing:
                raise KeyError(f"label_map is missing classes {sorted(missing)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        return image, self.label_map[label]


def indices_for_classes(dataset, classes):
    """Indices of every sample in `dataset` whose label is in `classes`."""
    targets = np.asarray(dataset.targets)
    return np.where(np.isin(targets, [int(c) for c in classes]))[0]


def make_loader(dataset, batch_size, shuffle, workers=0):
    """A DataLoader with the reproducibility settings this project expects."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=worker_init_fn if workers else None,
        pin_memory=False,
    )
