import argparse

from src import config


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="CS558 IRL pipeline entry point. Pick a subcommand.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    sub.add_parser("pick-place", help="M1 classical pick-and-place demo (PyBullet GUI).")
    sub.add_parser("init-dev", help="Init/dev sandbox scene.")

    train_p = sub.add_parser("train", help="M2: train PPO residual policy (headless).")
    train_p.add_argument(
        "--mode", choices=["hybrid", "rl_only"], default="hybrid",
        help="Policy variant: hybrid (PD + bounded residual) or rl_only (pure RL, no PD).",
    )
    train_p.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override PPO_TOTAL_TIMESTEPS from config (e.g. 200000 for a diagnostic run).",
    )
    train_p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel collector workers (default: PPO_NUM_COLLECTOR_WORKERS from config).",
    )
    train_p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (seeds PyTorch, NumPy, Python random).",
    )

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
    eval_p.add_argument(
        "--hybrid-model", default=None,
        help="Path to hybrid .pt (default: results/m2/models/final_model.pt).",
    )
    eval_p.add_argument(
        "--rl-only-model", default=None,
        help="Path to rl_only .pt (default: auto-discover results/m2/models_rl_only/final_model.pt).",
    )
    eval_p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel eval workers (default: EVAL_NUM_WORKERS from config).",
    )

    demo_p = sub.add_parser(
        "residual-demo",
        help="M2: full pick-and-place using the trained residual policy.",
    )
    demo_p.add_argument(
        "--mode", choices=["hybrid", "planner_only", "rl_only"], default="hybrid",
        help="Control mode: hybrid (PD+residual), planner_only (PD only, residual zeroed), rl_only.",
    )
    demo_p.add_argument(
        "--perturb-xy", type=float, default=None,
        help="Cube XY perturbation range in meters (default: PERTURB_XY_RANGE from config).",
    )
    demo_p.add_argument(
        "--no-retry", action="store_true", default=False,
        help="Disable grasp retry (single attempt, matches eval framework behavior).",
    )

    return parser


def main():
    args = _build_parser().parse_args()

    if args.command == "pick-place":
        from src.demo.pick_place import run_pick_place_demo
        run_pick_place_demo()
    elif args.command == "init-dev":
        from src.demo.init_dev import run_init_dev
        run_init_dev()
    elif args.command == "train":
        from src.rl.train import train
        train(mode=args.mode, total_timesteps=args.total_timesteps,
              num_workers=args.workers, seed=args.seed)
    elif args.command == "eval":
        from src.rl.evaluate import run_evaluation
        run_evaluation(
            model_path=args.hybrid_model,
            rl_only_model_path=args.rl_only_model,
            verbose_episodes=args.verbose,
            num_workers=args.workers,
        )
    elif args.command == "residual-demo":
        from src.demo.pick_place_residual import run_pick_place_with_residual
        run_pick_place_with_residual(
            mode=args.mode,
            perturb_xy_range=args.perturb_xy,
            allow_grasp_retry=not args.no_retry,
        )


if __name__ == "__main__":
    main()
