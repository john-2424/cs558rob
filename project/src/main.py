import argparse

from src import config


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="CS558 IRL pipeline entry point. Pick a subcommand.",
    )
    sub = parser.add_subparsers(dest="mode", required=True, metavar="<mode>")

    sub.add_parser("pick-place", help="M1 classical pick-and-place demo (PyBullet GUI).")
    sub.add_parser("init-dev", help="Init/dev sandbox scene.")
    sub.add_parser("train", help="M2: train PPO residual policy (headless).")

    eval_p = sub.add_parser("eval", help="M2: evaluate planner-only / hybrid / rl-only.")
    eval_default = bool(getattr(config, "EVAL_VERBOSE_EPISODES", False))
    verbose_group = eval_p.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "--verbose", dest="verbose", action="store_true", default=eval_default,
        help="Print per-episode diagnostic line (default from config: %(default)s).",
    )
    verbose_group.add_argument(
        "--quiet", dest="verbose", action="store_false",
        help="Suppress per-episode diagnostics.",
    )

    sub.add_parser(
        "residual-demo",
        help="M2: full pick-and-place using the trained residual policy.",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    if args.mode == "pick-place":
        from src.demo.pick_place import run_pick_place_demo
        run_pick_place_demo()
    elif args.mode == "init-dev":
        from src.demo.init_dev import run_init_dev
        run_init_dev()
    elif args.mode == "train":
        from src.rl.train import train
        train()
    elif args.mode == "eval":
        from src.rl.evaluate import run_evaluation
        run_evaluation(verbose_episodes=args.verbose)
    elif args.mode == "residual-demo":
        from src.demo.pick_place_residual import run_pick_place_with_residual
        run_pick_place_with_residual()


if __name__ == "__main__":
    main()
