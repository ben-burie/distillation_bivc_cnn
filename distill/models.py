"""The teacher and student networks, and the growable classifier head."""

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class IncrementalCNN(nn.Module):
    """A backbone (`features`) plus a linear head (`fc2`) that can grow.

    Subclasses provide `features(x)` and set `self.fc2`.
    """

    def features(self, x):
        raise NotImplementedError

    @property
    def output_layer(self):
        """The final Linear layer, under the name the calibration code expects."""
        return self.fc2

    def expand_head(self, num_classes):
        """Grow fc2 to `num_classes` outputs, keeping the weights learned so far."""
        old = self.fc2
        if num_classes <= old.out_features:
            return
        new = nn.Linear(old.in_features, num_classes).to(old.weight.device)
        with torch.no_grad():
            new.weight[: old.out_features] = old.weight
            new.bias[: old.out_features] = old.bias
        self.fc2 = new

    def forward(self, x):
        return self.fc2(self.features(x))


class SimpleCNN(IncrementalCNN):
    """The student: two conv blocks, ~0.6M parameters."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))


class TeacherCNN(IncrementalCNN):
    """The teacher: three conv blocks, wider throughout."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(256 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8  -> 4x4
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def frozen_copy(model):
    """A detached eval-mode clone, used as the anti-forgetting teacher."""
    if model is None:
        return None
    clone = copy.deepcopy(model)
    clone.eval()
    for p in clone.parameters():
        p.requires_grad_(False)
    return clone
