"""Accuracy on a loader, and the per-task breakdown."""

import logging

import torch

log = logging.getLogger(__name__)


@torch.no_grad()
def accuracy(model, loader, device):
    """Top-1 accuracy in percent; 0.0 for an empty loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = model(images).max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    if total == 0:
        log.warning("evaluated on an empty loader")
        return 0.0
    return 100.0 * correct / total


def evaluate(model, loader, device, name="model"):
    """Accuracy, logged with a label."""
    acc = accuracy(model, loader, device)
    log.info("%s test accuracy: %.2f%%", name, acc)
    return acc


def evaluate_per_task(model, task_loaders, device):
    """Accuracy on each task's own test set, oldest task first."""
    return [accuracy(model, loader, device) for loader in task_loaders]


def format_pct(value):
    """`None` reads as n/a, so a split with no samples on one side stays legible."""
    return f"{value:.1f}%" if value is not None else "n/a"


@torch.no_grad()
def evaluate_old_new(model, loader, device, num_old):
    """Accuracy overall, on the classes this task added, and on everything before them.

    The split is on the remapped label: slots below `num_old` were already in the head
    when the task started.  Either side comes back as None when it has no samples, which
    is the first task's case for `old_acc` and an exhausted replay memory's for it too.
    """
    model.eval()
    correct = total = new_correct = new_total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        hit = model(images).argmax(1) == labels
        correct += hit.sum().item()
        total += labels.size(0)
        new_mask = labels >= num_old
        new_correct += hit[new_mask].sum().item()
        new_total += int(new_mask.sum().item())

    old_total = total - new_total
    return {
        "acc": 100.0 * correct / total if total else 0.0,
        "new_acc": 100.0 * new_correct / new_total if new_total else None,
        "old_acc": 100.0 * (correct - new_correct) / old_total if old_total else None,
    }
