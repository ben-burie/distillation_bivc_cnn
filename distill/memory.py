"""The replay exemplar memory.

Two budgeting schemes.  A fixed *total* budget is split evenly over every class
seen so far, so the memory shrinks per class as the number of tasks grows
(iCaRL-style).  A fixed *per-class* budget keeps the same number of exemplars for
every class however many arrive, which is what the bias-correction split wants: it
needs to hold the same number of images out of every class, and cannot do that if
the per-class count is still shrinking underneath it.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from .data import indices_for_classes, make_loader

log = logging.getLogger(__name__)


class ExemplarMemory:
    def __init__(self, size=2000, selection="herding", per_class_size=None):
        if size < 0:
            raise ValueError(f"memory size must be >= 0, got {size}")
        if per_class_size is not None and per_class_size < 0:
            raise ValueError(f"per-class size must be >= 0, got {per_class_size}")
        if selection not in ("herding", "random"):
            raise ValueError(f"unknown selection strategy {selection!r}")
        self.size = size
        self.per_class_size = per_class_size
        self.selection = selection
        self.per_class = {}  # original class label -> list of train indices

    @property
    def enabled(self):
        return bool(self.per_class_size) if self.per_class_size is not None else self.size > 0

    def __len__(self):
        return len(self.indices())

    def indices(self):
        return [i for idxs in self.per_class.values() for i in idxs]

    def update(self, train_eval_set, new_classes, device, model=None,
               batch_size=256, workers=0, allowed=None):
        """Add `new_classes` to the memory and rebalance the budget.

        `allowed` restricts herding to a given set of indices.  The incremental
        experiment passes the task's training split, so an image held out for the
        bias-correction or validation set never becomes an exemplar and therefore never
        turns into training data at the next task.  (The voice-command pipeline herds
        over every clip of the new class; restricting it here keeps the held-out sets
        held out for the whole run, which matters more when a run has twenty tasks
        rather than one increment.)
        """
        if not self.enabled:
            return
        if self.per_class_size is not None:
            budget = self.per_class_size
        else:
            num_classes = len(self.per_class) + len(new_classes)
            budget = max(self.size // max(num_classes, 1), 1)

        for c in new_classes:
            pool = indices_for_classes(train_eval_set, [c])
            if allowed is not None:
                pool = pool[np.isin(pool, allowed)]
            if len(pool) == 0:
                log.warning("no training samples for class %s, skipping it", c)
                continue
            if self.selection == "herding" and model is not None:
                chosen = self._herd(train_eval_set, pool, budget, model, device,
                                    batch_size, workers)
            else:
                chosen = pool[torch.randperm(len(pool))[:budget].numpy()]
            self.per_class[int(c)] = [int(i) for i in chosen]

        if self.per_class_size is None:
            # Only the shared-budget scheme shrinks; a per-class budget is already met.
            for c in self.per_class:  # trim the classes that were already stored
                self.per_class[c] = self.per_class[c][:budget]

    @torch.no_grad()
    def _herd(self, dataset, pool, budget, model, device, batch_size, workers):
        """Greedily pick the exemplars whose running mean tracks the class mean."""
        model.eval()
        loader = make_loader(Subset(dataset, pool.tolist()), batch_size,
                             shuffle=False, workers=workers)
        feats = torch.cat([model.features(images.to(device)).cpu()
                           for images, _ in loader])
        feats = F.normalize(feats, dim=1)
        mu = F.normalize(feats.mean(0, keepdim=True), dim=1)

        selected, running = [], torch.zeros_like(mu)
        for k in range(min(budget, len(pool))):
            dist = (mu - (running + feats) / (k + 1)).pow(2).sum(1)
            dist[selected] = float("inf")  # never pick the same example twice
            best = int(dist.argmin())
            selected.append(best)
            running = running + feats[best]
        return pool[selected]

    def summary(self):
        scheme = (f"{self.per_class_size}/class" if self.per_class_size is not None
                  else f"{self.size} total")
        return f"{len(self)} exemplars over {len(self.per_class)} classes ({scheme})"
