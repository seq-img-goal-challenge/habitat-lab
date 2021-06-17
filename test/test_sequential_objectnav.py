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
    from habitat.tasks.sequential_nav.sequential_nav import SequentialSuccess, \
                                                            DistanceToNextGoal, \
                                                            SequentialSPL, \
                                                            SequentialProgress

    assert registry.get_dataset("SequentialObjectNav-v0") is SequentialObjectNavDatasetV0
    assert registry.get_task("SequentialObjectNav-v0") is SequentialObjectNavTask
    assert registry.get_sensor("SequentialObjectGoalCategorySensor") \
            is SequentialObjectGoalCategorySensor
    assert registry.get_sensor("SequentialObjectGoalAppearanceSensor") \
            is SequentialObjectGoalAppearanceSensor
    assert registry.get_measure("SequentialSuccess") is SequentialSuccess
    assert registry.get_measure("DistanceToNextGoal") is DistanceToNextGoal
    assert registry.get_measure("SequentialSPL") is SequentialSPL
    assert registry.get_measure("SequentialProgress") is SequentialProgress


def test_generate_dataset():
    from habitat.datasets.sequential_objectnav.sequential_objectnav_generator \
            import generate_sequential_objectnav_dataset

    generate_sequential_objectnav_dataset("configs/tasks/pointnav_gibson.yaml", [
        "DATASET.TYPE", "SequentialObjectNav-v0",
        "DATASET.DATA_PATH", "data/datasets/sequential_objectnav/testgen/{split}/{split}.json.gz",
    ], 10, 3, 8, 5, 1.0, "YAXIS", "OVERRIDE",
    "data/scene_datasets/gibson/", "data/object_datasets/test_objects/", 123456)


def test_task():
    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml", [
        "SIMULATOR.SCENE", "data/scene_datasets/gibson/Ackermanville.glb",
        "DATASET.TYPE", "SequentialObjectNav-v0",
        "DATASET.DATA_PATH", "data/datasets/sequential_objectnav/testgen/{split}/{split}.json.gz",
        "TASK.TYPE", "SequentialObjectNav-v0",
    ])
    cfg.defrost()
    cfg.TASK.SENSORS = ["SEQUENTIAL_OBJECTGOAL_CATEGORY", "SEQUENTIAL_OBJECTGOAL_APPEARANCE"]
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_CATEGORY = habitat.Config()
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_CATEGORY.TYPE = "SequentialObjectGoalCategorySensor"
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_CATEGORY.SEQUENTIAL_MODE = "FULL"
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_CATEGORY.PADDING_VALUE = -1
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE = habitat.Config()
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.TYPE = "SequentialObjectGoalAppearanceSensor"
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.SEQUENTIAL_MODE = "MYOPIC"
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.PADDING_VALUE = -1
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.OUT_OF_CONTEXT = False
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.OUT_OF_CONTEXT_POS = [0.0, 50.0, 0.0]
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.MAX_VIEW_DISTANCE = 2.0
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.MIN_VIEW_DISTANCE = 0.5
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.ISLAND_RADIUS = 0.2
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.RANDOM_OBJECT_ORIENTATION = "DISABLE"
    cfg.TASK.SEQUENTIAL_OBJECTGOAL_APPEARANCE.NUM_VIEWS = 5
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_NEXT_GOAL", "SEQUENTIAL_SUCCESS", "SEQUENTIAL_SPL"]
    cfg.TASK.DISTANCE_TO_NEXT_GOAL = habitat.Config()
    cfg.TASK.DISTANCE_TO_NEXT_GOAL.TYPE = "DistanceToNextGoal"
    cfg.TASK.SEQUENTIAL_SUCCESS = habitat.Config()
    cfg.TASK.SEQUENTIAL_SUCCESS.TYPE = "SequentialSuccess"
    cfg.TASK.SEQUENTIAL_SUCCESS.SUCCESS_DISTANCE = 1.0
    cfg.TASK.SEQUENTIAL_SPL = habitat.Config()
    cfg.TASK.SEQUENTIAL_SPL.TYPE = "SequentialSPL"
    cfg.freeze()

    env = habitat.Env(cfg)
    obs = env.reset()
    m = env.task.measurements.get_metrics()
    assert ["objectgoal_appearance", "objectgoal_category", "rgb"] == sorted(obs.keys())
    assert ["distance_to_next_goal", "sequential_spl", "sequential_success"] == sorted(m.keys())
