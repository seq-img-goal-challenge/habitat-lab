import os
import logging

os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

import pytest
import habitat
habitat.logger.setLevel(logging.ERROR)


def test_registration():
    from habitat.core.registry import registry
    from habitat.datasets.sequential_objectnav.sequential_objectnav_dataset \
            import SequentialObjectNavDatasetV0
    from habitat.tasks.sequential_nav.sequential_objectnav \
            import SequentialObjectNavTask, \
                   SequentialObjectGoalCategorySensor, \
                   SequentialObjectGoalAppearanceSensor

    assert registry.get_dataset("SequentialObjectNav-v0") is SequentialObjectNavDatasetV0
    assert registry.get_task("SequentialObjectNav-v0") is SequentialObjectNavTask
    assert registry.get_sensor("SequentialObjectGoalCategorySensor") \
            is SequentialObjectGoalCategorySensor
    assert registry.get_sensor("SequentialObjectGoalAppearanceSensor") \
            is SequentialObjectGoalAppearanceSensor


def test_generate_dataset():
    from habitat.datasets.sequential_objectnav.sequential_objectnav_generator \
            import generate_sequential_objectnav_dataset

    generate_sequential_objectnav_dataset("configs/tasks/pointnav_gibson.yaml", [
        "SIMULATOR.AGENT_0.SENSORS", "['DEPTH_SENSOR']",
        "DATASET.TYPE", "SequentialObjectNav-v0",
        "DATASET.DATA_PATH", "data/datasets/sequential_objectnav/testgen/{split}/{split}.json.gz",
    ], 10, 2, 3, 5, "VERTICAL", "OVERRIDE",
    "gibson", "shapenet_core_v2", 123685)


def test_task():
    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml", [
        "SIMULATOR.SCENE", "data/scene_datasets/gibson/Ackermanville.glb",
        "DATASET.TYPE", "SequentialObjectNav-v0",
        "DATASET.DATA_PATH", "data/datasets/sequential_objectnav/testgen/{split}/{split}.json.gz",
        "TASK.TYPE", "SequentialObjectNav-v0",
        "TASK.POSSIBLE_ACTIONS", "['FOUND', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']",
        "TASK.SENSORS", "['SEQUENTIAL_OBJECTGOAL_CATEGORY', 'SEQUENTIAL_OBJECTGOAL_APPEARANCE']",
        "TASK.MEASUREMENTS", "['DISTANCE_TO_NEXT_GOAL', 'SEQUENTIAL_SUCCESS', 'SEQUENTIAL_SPL']"
    ])

    env = habitat.Env(cfg)
    obs = env.reset()
    m = env.task.measurements.get_metrics()
    assert ["objectgoal_appearance", "objectgoal_category", "rgb"] == sorted(obs.keys())
    assert ["distance_to_next_goal", "seq_spl", "seq_success"] == sorted(m.keys())
