from src import config


def main():
    if config.RUN_M2_TRAINING:
        from src.rl.train import train
        train()
    elif config.RUN_M2_EVALUATION:
        from src.rl.evaluate import run_evaluation
        run_evaluation()
    elif config.RUN_M2_RESIDUAL_DEMO:
        from src.demo.pick_place_residual import run_pick_place_with_residual
        run_pick_place_with_residual()
    elif config.RUN_PICK_PLACE_DEMO:
        from src.demo.pick_place import run_pick_place_demo
        run_pick_place_demo()
    else:
        from src.demo.init_dev import run_init_dev
        run_init_dev()


if __name__ == "__main__":
    main()