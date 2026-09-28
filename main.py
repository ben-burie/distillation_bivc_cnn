#!/usr/bin/env python
"""Run the distillation pipeline.  See `python main.py --help`.

The implementation lives in the `distill` package, one module per stage:
config, data, models, losses, train, evaluate, memory, metrics, experiments.
"""

import sys

from distill.cli import main

if __name__ == "__main__":
    sys.exit(main())
