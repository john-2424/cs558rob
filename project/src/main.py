from src import config

from src.demo.init_dev import run_init_dev
from src.demo.pick_place import run_pick_place_demo


def main():
    if config.RUN_PICK_PLACE_DEMO:
        run_pick_place_demo()
    else:
        run_init_dev()


if __name__ == "__main__":
    main()