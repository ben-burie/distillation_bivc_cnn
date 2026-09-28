"""Command line parsing and validation."""

import argparse

from .bias_correction import MODES as BIC_MODES
from .data import DATASETS
from .train import LOSS_MODES

BIC_CHOICES = (*BIC_MODES, "none")

DESCRIPTION = """Distillation + exemplar replay + bias correction, on CIFAR-10/100.

Two modes:

  joint        the original setup: train the teacher on the whole dataset with
               hard labels, then distill it into the student.

  incremental  class-incremental learning: the classes arrive in groups (5 at a
               time by default).  One network is grown a head-slot at a time and
               trained in two stages per task --

                 stage 1  the iCaRL / LwF.MC objective against a frozen copy of
                          the previous task's network, over the new classes plus
                          a replay memory of old exemplars
                 stage 2  BiC: fit a calibration on a class-balanced held-out set
                          with the network frozen, then fold it into the head

               Three calibrations are available: scalar alpha/beta on the new-class
               logits (Wu et al.), one additive offset per class, or those offsets
               plus a shared alpha.  Each is folded into the output layer, so the
               corrected network stays an ordinary classifier.

Examples:
  python main.py                                        # CIFAR-100, 5 classes/task, BiC
  python main.py --classes-per-task 10 --epochs 10
  python main.py --bic-mode vector_alpha                # vector scaling with a shared alpha
  python main.py --bic-mode none                        # stage 1 only, the ablation
  python main.py --replay-ce-weight 0                   # iCaRL as specified, no CE term
  python main.py --incremental-loss lwf                 # softmax LwF instead of iCaRL
  python main.py --exemplars-per-class 0 --bic-mode none  # naive fine-tuning baseline
  python main.py --mode joint --dataset cifar10         # original behaviour
  python main.py --num-tasks 2 --epochs 1 --save-dir runs/smoke
"""


def build_parser():
    p = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    run = p.add_argument_group("run")
    run.add_argument("--mode", choices=["incremental", "joint"], default="incremental")
    run.add_argument("--dataset", choices=sorted(DATASETS), default="cifar100")
    run.add_argument("--root", default="./data")
    run.add_argument("--no-download", action="store_true",
                     help="never fetch the dataset; fail if it is not in --root")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="auto",
                     help="auto (default), cpu, cuda, cuda:1, ...")
    run.add_argument("--deterministic", action="store_true",
                     help="force deterministic cuDNN kernels (slower)")
    run.add_argument("--save-dir", default=None,
                     help="write checkpoints, metrics and a log file here")
    run.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    task = p.add_argument_group("incremental tasks")
    task.add_argument("--classes-per-task", type=int, default=5)
    task.add_argument("--num-tasks", type=int, default=0,
                      help="only run the first N tasks (0 = all); handy for quick runs")
    task.add_argument("--arch", choices=["teacher", "student"], default="teacher",
                      help="which backbone the single incremental network uses "
                           "(default: teacher, the wider three-block CNN)")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=5,
                     help="stage-1 epochs per task (and per model in joint mode)")
    opt.add_argument("--teacher-epochs", type=int, default=None,
                     help="joint mode only; defaults to --epochs")
    opt.add_argument("--student-epochs", type=int, default=None,
                     help="joint mode only; defaults to --epochs")
    opt.add_argument("--batch-size", type=int, default=64)
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--grad-clip", type=float, default=0.0,
                     help="clip gradients to this norm (0 = off)")
    opt.add_argument("--no-augment", action="store_true")
    opt.add_argument("--workers", type=int, default=0)

    kd = p.add_argument_group("distillation")
    kd.add_argument("--incremental-loss", choices=sorted(LOSS_MODES), default="icarl",
                    help="stage-1 objective: 'icarl' is the per-node BCE of iCaRL/LwF.MC, "
                         "'lwf' the softmax KL over the old logits (default: icarl)")
    kd.add_argument("--replay-ce-weight", type=float, default=1.0,
                    help="weight of the cross-entropy term added to the iCaRL loss; 0 is "
                         "iCaRL as specified, which supervises the old rows only through "
                         "the teacher (default: 1.0)")
    kd.add_argument("--T", type=float, default=4.0, help="distillation temperature")
    kd.add_argument("--alpha", type=float, default=0.5,
                    help="joint mode: weight of the teacher's soft targets in the student loss")
    kd.add_argument("--lwf-lambda", type=float, default=1.0,
                    help="--incremental-loss lwf only: weight of the old-model term")
    kd.add_argument("--lwf-scale", action="store_true",
                    help="scale --lwf-lambda by the fraction of classes already seen")

    mem = p.add_argument_group("replay memory")
    mem.add_argument("--memory-scheme", choices=["per-class", "total"], default="per-class",
                     help="'per-class' keeps --exemplars-per-class images for every class; "
                          "'total' splits --memory-size over the classes seen so far, so "
                          "the per-class count shrinks as tasks arrive (default: per-class)")
    mem.add_argument("--exemplars-per-class", type=int, default=20,
                     help="per-class scheme: exemplars kept for each class (0 = no replay)")
    mem.add_argument("--memory-size", type=int, default=2000,
                     help="total scheme: exemplars kept across all seen classes (0 = none)")
    mem.add_argument("--memory-select", choices=["herding", "random"], default="herding")

    bic = p.add_argument_group("bias correction (stage 2)")
    bic.add_argument("--bic-mode", choices=sorted(BIC_CHOICES), default="scalar",
                     # ASCII only: --help goes to stdout, which is cp1252 under a redirect
                     # on Windows and cannot encode the alpha/beta these names stand for.
                     help="'scalar' fits alpha/beta on the new-class logits, 'vector' one "
                          "offset per class, 'vector_alpha' those offsets plus a shared "
                          "alpha, 'none' skips stage 2 (default: scalar)")
    bic.add_argument("--bic-epochs", type=int, default=200,
                     help="epochs for the stage-2 fit (default: 200)")
    bic.add_argument("--bic-lr", type=float, default=1e-3,
                     help="learning rate for the stage-2 fit (default: 1e-3)")
    bic.add_argument("--bic-val-per-class", type=int, default=5,
                     help="images held out of every seen class for the balanced "
                          "calibration set (default: 5)")
    bic.add_argument("--old-val-per-class", type=int, default=15,
                     help="non-exemplar images drawn from every old class for the "
                          "per-epoch validation set (default: 15)")
    bic.add_argument("--new-holdout-frac", type=float, default=0.2,
                     help="fraction of each new class held out of stage-1 training, from "
                          "which the calibration and validation images are taken "
                          "(default: 0.2)")
    return p


def validate(args, parser=None):
    """Fill in the derived defaults and reject impossible combinations."""
    fail = parser.error if parser else _raise

    args.teacher_epochs = args.epochs if args.teacher_epochs is None else args.teacher_epochs
    args.student_epochs = args.epochs if args.student_epochs is None else args.student_epochs

    if args.classes_per_task < 1:
        fail("--classes-per-task must be at least 1")
    if args.num_tasks < 0:
        fail("--num-tasks must be 0 (all tasks) or positive")
    if args.batch_size < 1:
        fail("--batch-size must be at least 1")
    if min(args.epochs, args.teacher_epochs, args.student_epochs) < 0:
        fail("epoch counts must not be negative")
    if args.lr <= 0:
        fail("--lr must be positive")
    if args.T <= 0:
        fail("--T must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        fail("--alpha must be between 0 and 1")
    if args.lwf_lambda < 0:
        fail("--lwf-lambda must not be negative")
    if args.replay_ce_weight < 0:
        fail("--replay-ce-weight must not be negative")
    if args.memory_size < 0:
        fail("--memory-size must not be negative")
    if args.exemplars_per_class < 0:
        fail("--exemplars-per-class must not be negative")
    if args.workers < 0:
        fail("--workers must not be negative")
    if args.grad_clip < 0:
        fail("--grad-clip must not be negative")
    if args.bic_epochs < 1:
        fail("--bic-epochs must be at least 1")
    if args.bic_lr <= 0:
        fail("--bic-lr must be positive")
    if args.bic_val_per_class < 0:
        fail("--bic-val-per-class must not be negative")
    if args.old_val_per_class < 0:
        fail("--old-val-per-class must not be negative")
    if not 0.0 < args.new_holdout_frac < 1.0:
        fail("--new-holdout-frac must be between 0 and 1, exclusive")

    # `None` is what the memory takes to mean "share one budget out"; the per-class
    # scheme hands it the count instead.  Resolving it here keeps the branch in one place.
    if args.memory_scheme == "total":
        args.exemplars_per_class = None

    if args.mode == "incremental" and args.bic_mode != "none":
        if args.exemplars_per_class is None:
            fail("--bic-mode needs a per-class memory: the calibration set holds the same "
                 "number of images out of every class, which --memory-scheme total cannot "
                 "guarantee as its per-class count shrinks.  Use --memory-scheme per-class, "
                 "or --bic-mode none.")
        if args.bic_val_per_class < 1:
            fail("--bic-mode needs --bic-val-per-class to be at least 1")
        if args.exemplars_per_class <= args.bic_val_per_class:
            fail(f"--exemplars-per-class ({args.exemplars_per_class}) must exceed "
                 f"--bic-val-per-class ({args.bic_val_per_class}), or the old classes have "
                 f"nothing left to replay once the calibration set is held out")
    return args


def _raise(message):
    raise ValueError(message)


def parse_args(argv=None):
    parser = build_parser()
    return validate(parser.parse_args(argv), parser)
