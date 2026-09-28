"""The three disjoint splits each incremental task trains on.

Every task is cut into:

  train    the new classes' training images plus the replay exemplars kept for the
           old ones -- what stage 1 optimises
  bic      a *class-balanced* held-out set, `bic_val_per_class` images from every
           class seen so far -- what stage 2 fits alpha/beta (or b) on
  val      non-exemplar images from the old classes plus held-out new ones -- the
           per-epoch signal, and the before/after readout for stage 2

No image appears in more than one of them.  The balance in `bic` is the whole point:
the correction is estimated from a set where the new classes have no sample-count
advantage, so what it measures is the head's bias and not the task's own skew.

A caveat worth stating: the old-class images in `val` are non-exemplar *training*
images.  They are unseen by this task -- they are not in `train` and not in the
memory -- but the model did train on them when their class was new.  The test-set
numbers in evaluate.py are the clean ones; `val` is the in-training signal.
"""

import logging

import numpy as np

from .data import indices_for_classes

log = logging.getLogger(__name__)


def partition_old_classes(memory, train_eval, bic_val_per_class, val_per_class, rng=None):
    """Split the old classes three ways, with no image appearing in more than one set.

    Returns (train, bic, val) as arrays of indices into the training set.

    Pass an `rng` to draw both splits at random.  Without one the split is a fixed slice
    of the herding order and a fixed slice of the sorted index order, both of which are
    the same in every run -- herding is a deterministic greedy over the previous task's
    features, so a seed sweep that leaves this deterministic re-trains on identical data
    every time.
    """
    train, bic, val = [], [], []

    for label, exemplars in memory.per_class.items():
        exemplars = list(exemplars)
        # Never surrender the last exemplar to the calibration set: a class with nothing
        # in `train` gets no replay at all, which is a different experiment.
        n_train = max(len(exemplars) - bic_val_per_class, 1)
        if rng is None:
            # Herding order is most-representative-first, so the head of the list trains.
            ex_train, ex_bic = exemplars[:n_train], exemplars[n_train:]
        else:
            shuffled = list(rng.permutation(exemplars))
            ex_train, ex_bic = shuffled[:n_train], shuffled[n_train:]
        train.extend(int(i) for i in ex_train)
        bic.extend(int(i) for i in ex_bic)

        used = set(exemplars)
        pool = [int(i) for i in indices_for_classes(train_eval, [label]) if i not in used]
        if rng is None:
            held_out = pool[:val_per_class]
        else:
            take = min(val_per_class, len(pool))
            held_out = [pool[i] for i in rng.choice(len(pool), take, replace=False)] \
                if pool else []
        val.extend(held_out)

    return np.asarray(train, dtype=np.int64), np.asarray(bic, dtype=np.int64), \
        np.asarray(val, dtype=np.int64)


def partition_new_classes(train_set, new_classes, bic_val_per_class, val_per_class,
                          holdout_frac=0.2, rng=None):
    """Split each new class into train / bic / val.

    `holdout_frac` of every new class is held out of training, mirroring the 80/20 split
    the voice-command pipeline uses.  The held-out pool then yields `bic_val_per_class`
    images for the calibration set and up to `val_per_class` for validation.

    Anything left over is *dropped* rather than returned to training.  Old classes can
    only ever contribute `val_per_class` images to `val` -- that is all the replay memory
    leaves unused -- so handing the new classes their entire 20% would put ~95 new images
    against 15 old ones per class on CIFAR, and the resulting val accuracy would track
    the new classes almost alone.  The split is the same shape as the voice pipeline's,
    where the two happened to match at the sizes involved.
    """
    train, bic, val = [], [], []

    for label in new_classes:
        pool = indices_for_classes(train_set, [label])
        if len(pool) == 0:
            log.warning("no training images for class %s", label)
            continue
        order = rng.permutation(len(pool)) if rng is not None else np.arange(len(pool))
        pool = pool[order]

        n_held = int(round(holdout_frac * len(pool)))
        # The calibration set is what stage 2 exists for; starve it last.
        n_held = max(n_held, min(bic_val_per_class, len(pool) - 1))
        cls_train, held = pool[: len(pool) - n_held], pool[len(pool) - n_held:]

        cls_bic = held[:bic_val_per_class]
        cls_val = held[bic_val_per_class: bic_val_per_class + val_per_class]

        train.extend(int(i) for i in cls_train)
        bic.extend(int(i) for i in cls_bic)
        val.extend(int(i) for i in cls_val)

    return np.asarray(train, dtype=np.int64), np.asarray(bic, dtype=np.int64), \
        np.asarray(val, dtype=np.int64)


def build_task_splits(train_set, train_eval, memory, new_classes, bic_val_per_class,
                      val_per_class, holdout_frac=0.2, rng=None):
    """The three index sets for one task, old and new classes combined.

    Old-class indices are drawn from `train_eval` (the un-augmented view) for the
    calibration and validation sets and from `train_set` for replay, which is why both
    datasets are passed; they index the same underlying images, so an index means the
    same picture in either.
    """
    old_train, old_bic, old_val = partition_old_classes(
        memory, train_eval, bic_val_per_class, val_per_class, rng)
    new_train, new_bic, new_val = partition_new_classes(
        train_set, new_classes, bic_val_per_class, val_per_class, holdout_frac, rng)

    train = np.concatenate([new_train, old_train]) if len(old_train) else new_train
    bic = np.concatenate([old_bic, new_bic]) if len(old_bic) else new_bic
    val = np.concatenate([old_val, new_val]) if len(old_val) else new_val

    log.info("  train: %d (%d new + %d replay)   bic: %d (%d old + %d new)   "
             "val: %d (%d old + %d new)",
             len(train), len(new_train), len(old_train),
             len(bic), len(old_bic), len(new_bic),
             len(val), len(old_val), len(new_val))
    if len(memory.per_class) and (len(old_bic) == 0 or len(new_bic) == 0):
        log.warning("  bias-correction set is missing one side - the correction will be "
                    "poorly determined.")

    return train, bic, val
