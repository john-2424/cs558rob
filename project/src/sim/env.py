import time
from dataclasses import dataclass

import pybullet as p
import pybullet_data

from src import config


@dataclass
class LoadedObjects:
    plane_id: int
    table_id: int
    cube_id: int


class BulletEnv:
    def __init__(self, gui: bool = config.GUI, time_step: float = config.TIME_STEP):
        self.gui = gui
        self.time_step = time_step
        self.client_id = None
        self.objects = None

    def connect(self) -> int:
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(connection_mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, config.GRAVITY)
        p.setTimeStep(self.time_step)
        p.setRealTimeSimulation(1 if config.USE_REALTIME else 0)

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.35,
                cameraYaw=50,
                cameraPitch=-28,
                cameraTargetPosition=[0.55, -0.05, 0.35],
            )

        return self.client_id

    def load_scene(self) -> LoadedObjects:
        plane_id = p.loadURDF("plane.urdf")

        table_orn = p.getQuaternionFromEuler(config.TABLE_BASE_ORN_EULER)
        table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=config.TABLE_BASE_POS,
            baseOrientation=table_orn,
            useFixedBase=True,
        )

        cube_orn = p.getQuaternionFromEuler(config.CUBE_BASE_ORN_EULER)
        cube_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=config.CUBE_BASE_POS,
            baseOrientation=cube_orn,
            useFixedBase=False,
        )

        self.objects = LoadedObjects(
            plane_id=plane_id,
            table_id=table_id,
            cube_id=cube_id,
        )
        return self.objects

    def step(self, num_steps: int = 1, sleep: bool = True) -> None:
        for _ in range(num_steps):
            p.stepSimulation()
            if self.gui and sleep and not config.USE_REALTIME:
                time.sleep(self.time_step)

    def disconnect(self) -> None:
        if self.client_id is not None:
            p.disconnect()
            self.client_id = None