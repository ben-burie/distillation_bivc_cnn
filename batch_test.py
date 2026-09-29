import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from distill.report import append_master_csv

seeds = [0, 1, 2]
modes = ["scalar", "vector"]
classes_per_task = [5, 10, 20, 50]

EPOCHS = 15
NUM_CLASSES = 100  # cifar100; the T-column count is derived from this
MASTER_CSV = "runs/master_accuracy.csv"
MAX_TASKS = max(NUM_CLASSES // cpt for cpt in classes_per_task)
TIMESTAMP = datetime.now().isoformat(timespec="seconds")

count = 1
total = len(modes) * len(seeds) * len(classes_per_task)

for mode in modes:
    for seed in seeds:
        for increment in classes_per_task:
            tag = f"seed{seed}_mode{mode}_increment{increment}"
            save_dir = Path("runs") / tag
            print(f"\n{'=' * 70}\nTest {count}/{total} starting: {tag}\n{'=' * 70}",
                  flush=True)

            result = subprocess.run([
                sys.executable, "main.py",
                "--seed", str(seed),
                "--bic-mode", mode,
                "--classes-per-task", str(increment),
                "--epochs", str(EPOCHS),
                "--save-dir", str(save_dir),
            ])

            if result.returncode != 0:
                print(f"Test {count}/{total} FAILED (exit {result.returncode}). Stopping.")
                sys.exit(1)

            metrics_path = save_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as fh:
                    results = json.load(fh)
                append_master_csv(
                    results, MASTER_CSV, name="model", max_tasks=MAX_TASKS,
                    extra={"timestamp": TIMESTAMP, "run": tag, "mode": mode,
                           "seed": seed, "classes_per_task": increment,
                           "epochs": EPOCHS},
                )
            else:
                print(f"  warning: {metrics_path} is missing, nothing appended")

            print(f"Test {count}/{total} finished: {tag}", flush=True)
            count += 1

print(f"\nAll {total} runs complete. Master table: {MASTER_CSV}")