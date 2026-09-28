"""Entry point: parse the arguments, set the run up, dispatch to an experiment."""

import logging
import os
import sys

from .config import parse_args
from .experiments import run_incremental, run_joint
from .utils import resolve_device, save_json, set_seed, setup_logging

log = logging.getLogger(__name__)

EXPERIMENTS = {"incremental": run_incremental, "joint": run_joint}


def main(argv=None):
    args = parse_args(argv)
    log_file = os.path.join(args.save_dir, "run.log") if args.save_dir else None
    setup_logging(args.log_level, log_file)

    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    log.info("mode=%s device=%s seed=%d", args.mode, device, args.seed)
    save_json(vars(args), args.save_dir, "config")

    try:
        EXPERIMENTS[args.mode](args, device)
    except KeyboardInterrupt:
        log.warning("interrupted, stopping early")
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
