"""The original experiment: train the teacher on everything, then distill.

Run it on its own with:  python -m distill.experiments.joint --dataset cifar10
"""

import logging

from ..data import build_datasets, make_loader
from ..evaluate import evaluate
from ..models import SimpleCNN, TeacherCNN, count_parameters
from ..train import train_model
from ..utils import save_checkpoint, save_json

log = logging.getLogger(__name__)


def run_joint(args, device):
    train_set, _, test_set, num_classes = build_datasets(
        args.dataset, args.root, augment=not args.no_augment,
        download=not args.no_download)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True,
                               workers=args.workers)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False,
                              workers=args.workers)

    teacher = TeacherCNN(num_classes).to(device)
    student = SimpleCNN(num_classes).to(device)
    log.info("teacher %.2fM parameters, student %.2fM parameters",
             count_parameters(teacher) / 1e6, count_parameters(student) / 1e6)

    log.info("Training teacher...")
    train_model(teacher, train_loader, device, epochs=args.teacher_epochs,
                lr=args.lr, grad_clip=args.grad_clip, tag="teacher")
    teacher_acc = evaluate(teacher, test_loader, device, name="Teacher")
    save_checkpoint(teacher, args.save_dir, "teacher")

    log.info("Distilling into student...")
    train_model(student, train_loader, device, epochs=args.student_epochs,
                lr=args.lr, teacher=teacher, T=args.T, alpha=args.alpha,
                grad_clip=args.grad_clip, tag="student")
    student_acc = evaluate(student, test_loader, device, name="Student")
    save_checkpoint(student, args.save_dir, "student")

    results = {"mode": "joint", "dataset": args.dataset,
               "teacher_accuracy": teacher_acc, "student_accuracy": student_acc}
    save_json(results, args.save_dir, "metrics")
    return results


if __name__ == "__main__":  # python -m distill.experiments.joint [options]
    import sys

    from ..cli import main

    raise SystemExit(main(["--mode", "joint", *sys.argv[1:]]))
