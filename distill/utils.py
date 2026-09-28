"""Device selection, seeding, logging and checkpoint helpers."""

import json
import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def setup_logging(level="INFO", log_file=None):
    """Send messages to stderr, and to `log_file` as well when given."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


def resolve_device(requested="auto"):
    """Turn --device into a real torch.device, falling back to the CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA was requested but is not available; running on the CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed, deterministic=False):
    """Seed every generator the pipeline draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Give every DataLoader worker its own reproducible seed."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(model, save_dir, name):
    """Write `model`'s weights to <save_dir>/<name>.pt; no-op without a dir."""
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    torch.save({"state_dict": model.state_dict(),
                "num_classes": model.fc2.out_features}, path)
    log.debug("saved checkpoint %s", path)
    return path


def save_json(payload, save_dir, name):
    """Write a metrics dict next to the checkpoints; no-op without a dir."""
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote %s", path)
    return path
