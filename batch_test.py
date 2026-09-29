import sys
import subprocess

seeds = [0, 1, 2]
modes = ["scalar", "vector"]
classes_per_task = [5, 10, 20, 50]

EPOCHS = 15
count = 1

for mode in modes:
    for seed in seeds:
        for increment in classes_per_task:
            print(f"Test {count} starting.")
            command = [
                sys.executable,
                "main.py",
                "--seed", str(seed),
                "--bic-mode", mode,
                "--classes-per-task", str(increment),
                "--epochs", str(EPOCHS),
                "--save-dir", f"runs/seed{seed}_mode{mode}_increment{increment}"
            ]
            result = subprocess.run(command)

            if result.returncode != 0:
                print(f"Test {count} FAILED. Exiting all tests.")
                sys.exit(0)

            print(f"Test {count} finished.")
            count+=1