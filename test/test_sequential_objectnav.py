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
    from habitat.datasets.sequential_objectnav.sequential_objectnav_dataset \
            import SequentialObjectNavDatasetV0
    from habitat.datasets.sequential_objectnav.sequential_objectnav_generator \
            import generate_sequential_objectnav_dataset, ObjectRotation, ExistBehavior
    from habitat.tasks.sequential_nav.sequential_objectnav \
            import SequentialObjectNavEpisode, SequentialObjectNavStep
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal

    CFG_PATH = "configs/tasks/sequential_objectnav.yaml"
    NUM_EPISODES = 3
    MIN_SEQ_LEN = 1
    MAX_SEQ_LEN = 3
    MAX_GOALS = 2
    NUM_RETRIES = 5
    SCENES_DIR = "data/scene_datasets/habitat-test-scenes"
    SCENE_EXT = ".glb"
    if not os.path.isdir(SCENES_DIR):
        pytest.skip(f"Test scenes '{SCENES_DIR}' not available.")
    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    generate_sequential_objectnav_dataset(CFG_PATH, [], SCENES_DIR, OBJECTS_DIR, 
                                          NUM_EPISODES, MIN_SEQ_LEN, MAX_SEQ_LEN,
                                          MAX_GOALS, ObjectRotation.FIXED,
                                          ExistBehavior.OVERRIDE, NUM_RETRIES, 123456)
    data_cfg = habitat.get_config(CFG_PATH).DATASET
    data_path = data_cfg.DATA_PATH.format(split=data_cfg.SPLIT)
    assert os.path.isfile(data_path)

    dataset = habitat.make_dataset(data_cfg.TYPE, config=data_cfg)
    assert isinstance(dataset, SequentialObjectNavDatasetV0)
    assert len(dataset.episodes) == NUM_EPISODES
    seen_id = set()
    for episode in dataset.episodes:
        assert isinstance(episode, SequentialObjectNavEpisode)
        assert episode.episode_id not in seen_id
        seen_id.add(episode.episode_id)
        assert episode.scene_id.endswith(SCENE_EXT)
        assert os.path.isfile(episode.scene_id)
        assert MIN_SEQ_LEN <= len(episode.steps) <= MAX_SEQ_LEN
        for step in episode.steps:
            assert isinstance(step, SequentialObjectNavStep)
            assert 0 < len(step.goals) <= MAX_GOALS
            for goal in step.goals:
                assert isinstance(goal, SpawnedObjectGoal)
                assert goal.object_template_id.endswith(OBJECT_EXT)
                assert os.path.isfile(goal.object_template_id)
                assert goal._spawned_object_id is None
                assert 0 < len(goal.view_points) <= 200



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
