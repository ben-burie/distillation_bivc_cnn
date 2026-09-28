"""Accuracy bookkeeping across tasks, and the end-of-run summary."""

import logging

log = logging.getLogger(__name__)

MODELS = ("teacher", "student")


class IncrementalMetrics:
    """Per-task accuracies for both models.

    seen_acc[name]  accuracy over every class seen so far, one entry per task
    task_acc[name]  one row per task: that model's accuracy on each task so far
    """

    def __init__(self, names=MODELS):
        self.names = tuple(names)
        self.seen_acc = {name: [] for name in self.names}
        self.task_acc = {name: [] for name in self.names}

    def record(self, name, seen_accuracy, per_task):
        self.seen_acc[name].append(seen_accuracy)
        self.task_acc[name].append(list(per_task))

    def average_incremental(self, name):
        accs = self.seen_acc[name]
        return sum(accs) / len(accs) if accs else 0.0

    def final(self, name):
        accs = self.seen_acc[name]
        return accs[-1] if accs else 0.0

    def forgetting(self, name):
        """Mean drop of each earlier task from its best accuracy to its last."""
        rows = self.task_acc[name]
        if len(rows) < 2:
            return None
        drops = [max(rows[t][i] for t in range(i, len(rows))) - rows[-1][i]
                 for i in range(len(rows) - 1)]
        return sum(drops) / len(drops)

    def as_dict(self):
        return {
            name: {
                "seen_accuracy": self.seen_acc[name],
                "per_task_accuracy": self.task_acc[name],
                "final": self.final(name),
                "average_incremental": self.average_incremental(name),
                "forgetting": self.forgetting(name),
            }
            for name in self.names
        }


def log_summary(metrics):
    log.info("\n================ summary ================")
    for name in metrics.names:
        accs = metrics.seen_acc[name]
        if not accs:
            log.info("%s: no results recorded", name)
            continue
        log.info("%s: accuracy on all seen classes after each task: %s",
                 name, " ".join(f"{a:.1f}" for a in accs))
        log.info("%s: final %.2f%%   average incremental %.2f%%",
                 name, metrics.final(name), metrics.average_incremental(name))
        forgetting = metrics.forgetting(name)
        if forgetting is not None:
            log.info("%s: average forgetting %.2f points", name, forgetting)
