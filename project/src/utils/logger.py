import json
from pathlib import Path


class TrajectoryLogger:
    def __init__(self):
        self.records = []

    def log(
        self,
        phase,
        waypoint_index,
        sim_step,
        q_target,
        q_actual,
        q_error,
        ee_pos,
        ee_euler,
    ):
        self.records.append(
            {
                "phase": phase,
                "waypoint_index": int(waypoint_index),
                "sim_step": int(sim_step),
                "q_target": list(map(float, q_target)),
                "q_actual": list(map(float, q_actual)),
                "q_error": list(map(float, q_error)),
                "ee_pos": list(map(float, ee_pos)),
                "ee_euler": list(map(float, ee_euler)),
            }
        )

    def save_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)