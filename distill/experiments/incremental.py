"""Class-incremental learning: distillation + exemplar replay + bias correction.

One network, grown a head-slot at a time, trained in two stages per task -- the same
shape as the voice-command pipeline this is ported from, with a CNN over images in
place of a classifier head over Whisper features:

  stage 1  the iCaRL / LwF.MC objective against a frozen copy of the previous task's
           network, over the new classes' images plus the replay memory
  stage 2  BiC: fit a small calibration on a class-balanced held-out set with the
           network frozen, then fold it into the output layer

The teacher/student pair the joint experiment uses is gone from here: the only teacher
an incremental task has is its own previous snapshot.  `--mode joint` still runs the
original two-network distillation.

Run it on its own with:  python -m distill.experiments.incremental --epochs 1
"""

import logging

import numpy as np

from ..bias_correction import apply_bias_correction
from ..data import RemappedSubset, build_datasets, indices_for_classes, make_loader
from ..evaluate import evaluate, evaluate_old_new, evaluate_per_task, format_pct
from ..memory import ExemplarMemory
from ..metrics import IncrementalMetrics, log_summary
from ..models import SimpleCNN, TeacherCNN, count_parameters, frozen_copy
from ..splits import build_task_splits
from ..train import train_model
from ..utils import save_checkpoint, save_json

log = logging.getLogger(__name__)

ARCHITECTURES = {"student": SimpleCNN, "teacher": TeacherCNN}

MODEL = "model"  # the single network's name in the metrics tables


def build_task_split(num_classes, classes_per_task, num_tasks, seed):
    """A fixed shuffled class order, cut into tasks.

    Returns the tasks and the map from the dataset's class id to its slot in
    the growing classifier head.
    """
    class_order = np.random.RandomState(seed).permutation(num_classes)
    label_map = {int(c): slot for slot, c in enumerate(class_order)}
    tasks = [class_order[i:i + classes_per_task]
             for i in range(0, num_classes, classes_per_task)]
    if num_tasks:
        tasks = tasks[:num_tasks]
    return tasks, label_map


def run_incremental(args, device):
    train_set, train_eval, test_set, num_classes = build_datasets(
        args.dataset, args.root, augment=not args.no_augment,
        download=not args.no_download)

    tasks, label_map = build_task_split(num_classes, args.classes_per_task,
                                        args.num_tasks, args.seed)
    log.info("%s: %d tasks x %d classes (class-order seed %d)",
             args.dataset, len(tasks), args.classes_per_task, args.seed)
    log.info("loss=%s  replay-ce=%.2f  bic=%s  memory=%s",
             args.incremental_loss, args.replay_ce_weight, args.bic_mode,
             f"{args.exemplars_per_class}/class" if args.exemplars_per_class is not None
             else f"{args.memory_size} total")

    model = None  # built with the first task's classes, grown after
    memory = ExemplarMemory(args.memory_size, args.memory_select,
                            per_class_size=args.exemplars_per_class)
    metrics = IncrementalMetrics(names=(MODEL,))

    seen_classes = []
    task_test_loaders = []  # one loader per task, for the breakdown
    bic_records = []

    for t, new_classes in enumerate(tasks):
        num_old = len(seen_classes)
        seen_classes.extend(int(c) for c in new_classes)
        num_seen = len(seen_classes)

        log.info("\n=== Task %d/%d: classes %s (%d seen in total) ===",
                 t + 1, len(tasks), list(new_classes), num_seen)

        # A fresh stream per task: herding is a deterministic greedy over the previous
        # task's features, so a split left un-randomised would draw the identical
        # calibration set in every run of a seed sweep.
        rng = np.random.RandomState(args.seed + t)
        train_idx, bic_idx, val_idx = build_task_splits(
            train_set, train_eval, memory, new_classes,
            args.bic_val_per_class, args.old_val_per_class,
            args.new_holdout_frac, rng)

        train_loader = _loader(args, train_set, train_idx, label_map, shuffle=True)
        # The calibration and validation sets go through the eval transform: a
        # correction fit on randomly cropped and flipped images would be fitting the
        # augmentation as much as the head.
        bic_loader = _loader(args, train_eval, bic_idx, label_map, shuffle=False)
        val_loader = _loader(args, train_eval, val_idx, label_map, shuffle=False)

        task_test_loaders.append(_subset_loader(args, test_set, new_classes, label_map))
        seen_test_loader = _subset_loader(args, test_set, seen_classes, label_map)

        # The snapshot of the previous task is the anti-forgetting teacher.
        old_model = frozen_copy(model) if num_old else None

        if model is None:
            model = ARCHITECTURES[args.arch](num_seen).to(device)
            log.info("%s: %.2fM parameters", args.arch, count_parameters(model) / 1e6)
        else:
            model.expand_head(num_seen)

        # Scaling by the share of old classes keeps the balance between remembering and
        # learning roughly constant as the head grows.  It only reaches the softmax LwF
        # term; the iCaRL objective has no such weight by construction.
        lwf_lambda = args.lwf_lambda * (num_old / num_seen) if args.lwf_scale \
            else args.lwf_lambda

        log.info("Stage 1: training on %d images...", len(train_idx))
        train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr,
                    old_model=old_model, num_old=num_old, T=args.T,
                    lwf_lambda=lwf_lambda, grad_clip=args.grad_clip, tag="stage1",
                    loss_mode=args.incremental_loss,
                    replay_ce_weight=args.replay_ce_weight, val_loader=val_loader)

        # Herd the new classes' exemplars with the freshly trained model, before the
        # correction is folded in -- the features herding scores are the same either
        # way, but this is the order the voice pipeline runs and it keeps stage 2 the
        # last thing that touches the weights.
        memory.update(train_eval, new_classes, device, model=model,
                      batch_size=args.batch_size, workers=args.workers,
                      allowed=train_idx)
        if memory.enabled:
            log.info("  replay memory: %s", memory.summary())

        record = _run_stage_two(args, model, bic_loader, val_loader, device,
                                num_old, num_seen)
        if record is not None:
            bic_records.append({"task": t + 1, **record})

        acc = evaluate(model, seen_test_loader, device,
                       name=f"Test (all {num_seen} seen classes)")
        per_task = evaluate_per_task(model, task_test_loaders, device)
        metrics.record(MODEL, acc, per_task)
        log.info("  per-task: %s",
                 "  ".join(f"T{i+1}:{a:.1f}" for i, a in enumerate(per_task)))
        save_checkpoint(model, args.save_dir, f"model_task{t+1}")

    log_summary(metrics)
    results = {"mode": "incremental", "dataset": args.dataset,
               "arch": args.arch, "loss": args.incremental_loss,
               "bic_mode": args.bic_mode,
               "tasks": [[int(c) for c in task] for task in tasks],
               "bias_correction": bic_records,
               **metrics.as_dict()}
    save_json(results, args.save_dir, "metrics")
    return results


def _run_stage_two(args, model, bic_loader, val_loader, device, num_old, num_seen):
    """Fit and fold the bias correction, reporting validation either side of it.

    Skipped on the first task, which has no old classes to be biased against, and when
    `--bic-mode none` turns the stage off for a baseline arm.
    """
    if args.bic_mode == "none" or num_old == 0:
        return None
    if len(bic_loader) == 0:
        log.warning("Stage 2 skipped: the bias-correction set is empty.")
        return None

    log.info("Stage 2: bias correction (%s) on %d held-out images...",
             args.bic_mode, len(bic_loader.dataset))
    before = evaluate_old_new(model, val_loader, device, num_old)
    record = apply_bias_correction(model, bic_loader, num_old, num_seen, device,
                                   mode=args.bic_mode, epochs=args.bic_epochs,
                                   lr=args.bic_lr)
    after = evaluate_old_new(model, val_loader, device, num_old)

    for label, val in (("before", before), ("after ", after)):
        log.info("  val %s | %.1f%% (new %s / old %s)", label, val["acc"],
                 format_pct(val["new_acc"]), format_pct(val["old_acc"]))
    record["val_before"] = before
    record["val_after"] = after
    return record


def _loader(args, dataset, indices, label_map, shuffle):
    return make_loader(RemappedSubset(dataset, indices, label_map),
                       args.batch_size, shuffle=shuffle, workers=args.workers)


def _subset_loader(args, dataset, classes, label_map):
    return _loader(args, dataset, indices_for_classes(dataset, classes), label_map,
                   shuffle=False)


if __name__ == "__main__":  # python -m distill.experiments.incremental [options]
    import sys

    from ..cli import main

    raise SystemExit(main(["--mode", "incremental", *sys.argv[1:]]))
