"""The training loop shared by every regime.

One loop covers them all, depending on what it is handed:

  teacher=None, old_model=None   plain cross-entropy
  teacher=...                    distillation from a separate, larger teacher
  old_model=..., loss_mode=lwf   plus softmax LwF over the old logits
  old_model=..., loss_mode=icarl the iCaRL / LwF.MC per-node BCE objective, which
                                 replaces cross-entropy rather than adding to it

`icarl` is what the incremental experiment runs by default.  The first task has no
previous snapshot to distil from, so it trains under plain cross-entropy however
`loss_mode` is set -- the same shape as the voice-command pipeline, whose base model
is trained with cross-entropy and whose increments are iCaRL.
"""

import logging

import torch
import torch.nn.functional as F

from .evaluate import evaluate_old_new, format_pct
from .losses import distillation_loss, icarl_distillation_loss, lwf_loss

log = logging.getLogger(__name__)

LOSS_MODES = ("icarl", "lwf")


def train_model(model, loader, device, epochs=5, lr=1e-3, teacher=None,
                old_model=None, num_old=0, T=4.0, alpha=0.5, lwf_lambda=1.0,
                grad_clip=0.0, tag="model", loss_mode="lwf", replay_ce_weight=0.0,
                val_loader=None):
    """Train `model` in place and return the mean loss of each epoch."""
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"unknown loss mode {loss_mode!r}; choose from {list(LOSS_MODES)}")
    if len(loader) == 0:
        log.warning("[%s] empty data loader, nothing to train on", tag)
        return []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if teacher is not None:
        teacher.eval()
    if old_model is not None:
        old_model.eval()
    has_old = old_model is not None and num_old > 0
    use_icarl = has_old and loss_mode == "icarl"
    use_lwf = has_old and loss_mode == "lwf" and lwf_lambda > 0

    history = []
    for epoch in range(epochs):
        model.train()
        running = {"loss": 0.0, "distill": 0.0, "class": 0.0, "replay": 0.0}
        correct = total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            old_logits = None
            if has_old:
                with torch.no_grad():
                    old_logits = old_model(images)

            if use_icarl:
                # The whole objective, not a term added to cross-entropy: the new-class
                # nodes get their hard targets from `classification`, the old ones from
                # the snapshot, and `replay` is the only softmax term in play.
                loss, distill, classification, replay = icarl_distillation_loss(
                    logits, old_logits, labels, num_old, replay_ce_weight)
                running["distill"] += distill.item()
                running["class"] += classification.item()
                running["replay"] += replay.item()
            else:
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    loss = distillation_loss(logits, teacher_logits, labels, T=T, alpha=alpha)
                else:
                    loss = F.cross_entropy(logits, labels)
                if use_lwf:
                    loss = loss + lwf_lambda * lwf_loss(logits, old_logits, num_old, T)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        n = len(loader)
        mean_loss = running["loss"] / n
        history.append(mean_loss)

        parts = [f"  [{tag}] epoch {epoch + 1}/{epochs}",
                 f"train {mean_loss:.4f}"]
        if use_icarl:
            parts.append(f"(d {running['distill'] / n:.4f} / c {running['class'] / n:.4f}"
                         f" / r {running['replay'] / n:.4f})")
        parts.append(f"/ {100 * correct / max(total, 1):.1f}%")

        if val_loader is not None and len(val_loader):
            val = evaluate_old_new(model, val_loader, device, num_old)
            parts.append(f"| val {val['acc']:.1f}% "
                         f"(new {format_pct(val['new_acc'])} / old {format_pct(val['old_acc'])})")
        log.info(" ".join(parts))

    return history
