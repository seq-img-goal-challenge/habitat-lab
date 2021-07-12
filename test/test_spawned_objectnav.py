import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

import glob
import os.path
import pytest
import numpy as np
import magnum as mn

import habitat
habitat.logger.setLevel(2)


def test_registration():
    from habitat.core.registry import registry
    from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset \
            import SpawnedObjectNavDatasetV0
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectNavTask, \
                                                    SpawnedObjectGoalCategorySensor, \
                                                    SpawnedObjectGoalAppearanceSensor
        
    assert registry.get_dataset("SpawnedObjectNav-v0") is SpawnedObjectNavDatasetV0
    assert registry.get_task("SpawnedObjectNav-v0") is SpawnedObjectNavTask
    assert registry.get_sensor("SpawnedObjectGoalCategorySensor") \
            is SpawnedObjectGoalCategorySensor
    assert registry.get_sensor("SpawnedObjectGoalAppearanceSensor") \
            is SpawnedObjectGoalAppearanceSensor


def test_create_object_pool():
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_object_pool

    obj_pool, category_index_map = create_object_pool("data/object_datasets/test_objects")
    expected = {"box": 3, "chair": 1, "sphere": 1, "donut": 1}
    assert sorted(obj_pool) == sorted(expected)
    assert all(len(tmpl_ids) == expected[cat] for cat, tmpl_ids in obj_pool.items())
    assert sorted(category_index_map) == sorted(expected)
    index_list = list(category_index_map.values())
    assert all(idx not in index_list[:i] for i, idx in enumerate(index_list))


def test_create_scene_pool():
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_scene_pool

    SCENE_DIR = "data/scene_datasets/gibson"
    SCENE_EXT = ".glb"
    if not os.path.isdir(SCENE_DIR):
        pytest.skip("Cannot locate gibson scene dataset")

    scene_pool = create_scene_pool(SCENE_DIR)
    expected = {entry.path for entry in os.scandir(SCENE_DIR)
                if entry.name.endswith(SCENE_EXT)}
    assert sorted(scene_pool) == sorted(expected)
    scene_pool_default_prefix = create_scene_pool("gibson")
    assert sorted(scene_pool_default_prefix) == sorted(expected)


def test_generate_episode():
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_object_pool, generate_spawned_objectnav_episode

    cfg = habitat.get_config().SIMULATOR
    cfg.defrost()
    cfg.SCENE = "data/scene_datasets/gibson/Dansville.glb"
    cfg.AGENT_0.SENSORS = ["DEPTH_SENSOR"]
    cfg.freeze()
    sim = habitat.sims.make_sim(cfg.TYPE, config=cfg)
    object_pool, category_index_map = create_object_pool("data/object_datasets/test_objects")

    SEED = 1234567
    sim.seed(SEED)

    rng = np.random.default_rng(SEED)
    for ep_idx, rot in enumerate(("DISABLE", "VERTICAL", "3D")):
        episode = generate_spawned_objectnav_episode(sim, object_pool, category_index_map,
                                                     "ep{}".format(ep_idx), rng, 2, rot)
        assert episode.episode_id == "ep{}".format(ep_idx)
        assert episode.scene_id == cfg.SCENE
        assert sim.island_radius(episode.start_position) > 0.2
        assert episode.object_category in {"box", "chair", "sphere", "donut"}
        assert episode.object_category_index == category_index_map[episode.object_category]
        assert 1 <= len(episode.goals) <= 2
        assert all(len(goal.view_points) > 0 for goal in episode.goals)
        assert all(np.isfinite(sim.geodesic_distance(episode.start_position, view_pt.position))
                   for goal in episode.goals for view_pt in goal.view_points)


def test_generate_dataset():
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import generate_spawned_objectnav_dataset

    PATH = "data/datasets/spawned_objectnav/test_gen_dataset/{split}/{split}.json.gz"
    EXTRA_CONFIG = ["SIMULATOR.AGENT_0.SENSORS", "['DEPTH_SENSOR']",
                    "DATASET.TYPE", "SpawnedObjectNav-v0",
                    "DATASET.DATA_PATH", PATH,
                    "DATASET.SPLIT", "test"]
    NUM_EPISODES = 3
    MAX_GOALS = 4
    OBJECT_ROT = "3D"
    SEED = 123456
    generate_spawned_objectnav_dataset("configs/tasks/pointnav.yaml", EXTRA_CONFIG,
                                       NUM_EPISODES, MAX_GOALS, OBJECT_ROT,
                                       "OVERRIDE", "gibson", "test_objects", SEED)
    assert os.path.isfile(PATH.format(split="test"))

    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml", EXTRA_CONFIG).DATASET
    dataset = habitat.make_dataset(cfg.TYPE, config=cfg)
    expected_categories = {"box", "sphere", "donut", "chair"}
    assert len(dataset.episodes) == NUM_EPISODES
    id_list = [episode.episode_id for episode in dataset.episodes]
    assert all(ep_id not in id_list[:i] for i, ep_id in enumerate(id_list))


def test_dataset():
    PATH = "data/datasets/spawned_objectnav/test_gen_dataset/{split}/{split}.json.gz"
    EXTRA_CONFIG = ["DATASET.TYPE", "SpawnedObjectNav-v0",
                    "DATASET.DATA_PATH", PATH,
                    "DATASET.SPLIT", "test"]
    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml", EXTRA_CONFIG).DATASET
    dataset = habitat.make_dataset(cfg.TYPE, config=cfg)

    all_tmpl_ids = dataset.get_objects_to_load()
    expected = set(glob.glob("./data/object_datasets/test_objects/**/*.object_config.json", recursive=True))
    assert all(tmpl_id in expected for tmpl_id in all_tmpl_ids)
    ep0_tmpl_ids = dataset.get_objects_to_load(dataset.episodes[0])
    assert all(tmpl_id in all_tmpl_ids for tmpl_id in ep0_tmpl_ids)

    category_map = dataset.get_object_category_map()
    expected = {"box", "chair", "sphere", "donut"}
    assert all(key in expected for key in category_map)
    index_list = list(category_map.values())
    assert all(cat_index not in index_list[:i] for i, cat_index in enumerate(index_list))
    max_cat_index = dataset.get_max_object_category_index()
    assert max_cat_index == max(index_list)


def test_category_sensor():
    sensor_cfg = habitat.Config()
    sensor_cfg.TYPE = "SpawnedObjectGoalCategorySensor"
    sensor_cfg.freeze()

    PATH = "data/datasets/spawned_objectnav/test_gen_dataset/{split}/{split}.json.gz"
    EXTRA_CONFIG = ["DATASET.TYPE", "SpawnedObjectNav-v0",
                    "DATASET.DATA_PATH", PATH,
                    "DATASET.SPLIT", "test"]
    data_cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml", EXTRA_CONFIG).DATASET
    dataset = habitat.make_dataset(data_cfg.TYPE, config=data_cfg)

    sensor_cls = habitat.registry.get_sensor(sensor_cfg.TYPE)
    sensor = sensor_cls(sensor_cfg, dataset)

    for episode in dataset.episodes:
        cat_index_arr = sensor.get_observation(episode)
        assert cat_index_arr.dtype == np.int64
        assert 0 <= cat_index_arr <= dataset.get_max_object_category_index()
        assert np.array(episode.object_category_index) == cat_index_arr


def test_appearance_sensor():
    import cv2
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode

    sensor_cfg = habitat.Config()
    sensor_cfg.TYPE = "SpawnedObjectGoalAppearanceSensor"
    sensor_cfg.OUT_OF_CONTEXT = True
    sensor_cfg.OUT_OF_CONTEXT_POS = [0.0, 50.0, 0.0]
    sensor_cfg.MAX_VIEW_DISTANCE = 2.0
    sensor_cfg.MIN_VIEW_DISTANCE = 0.5
    sensor_cfg.RANDOM_OBJECT_ROTATION = "DISABLE"
    sensor_cfg.NUM_VIEWS = 5
    sensor_cfg.freeze()
    sensor_cls = habitat.registry.get_sensor(sensor_cfg.TYPE)

    sim_cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml").SIMULATOR
    sim_cfg.defrost()
    sim_cfg.RGB_SENSOR.ORIENTATION = [-0.3, 0.0, 0.0]
    sim_cfg.DEPTH_SENSOR.ORIENTATION = [-0.3, 0.0, 0.0]
    sim_cfg.SCENE = "data/scene_datasets/gibson/Airport.glb"
    sim_cfg.freeze()
    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        tmpl_mngr = sim.get_object_template_manager()
        for path in glob.glob("data/object_datasets/test_objects/**/*.object_config.json", recursive=True):
            goal = SpawnedObjectGoal(position=sim.sample_navigable_point(),
                                     rotation=[0.0, 0.0, 0.0, 1.0],
                                     view_points=[],
                                     object_template_id=path)
            mngr_id, = tmpl_mngr.load_configs(goal.object_template_id)
            goal._spawned_object_id = sim.add_object(mngr_id)
            pos = [goal.position[0], goal.position[1] + 0.7, goal.position[2]]
            sim.set_translation(pos, goal._spawned_object_id)

            episode = SpawnedObjectNavEpisode(episode_id="ep0", scene_id=sim_cfg.SCENE,
                                              object_category="test", object_category_index=0,
                                              start_position=sim.sample_navigable_point(),
                                              start_rotation=[0.0, 0.0, 0.0, 1.0],
                                              goals=[goal])
            sensor = sensor_cls(sensor_cfg, sim)
            views = sensor.get_observation(episode)

            cv2.imshow("Views", np.concatenate([view for view in views], axis=1))
            cv2.waitKey(1000)
    cv2.destroyAllWindows()


def test_task():
    PATH = "data/datasets/spawned_objectnav/test_gen_dataset/{split}/{split}.json.gz"
    EXTRA_CONFIG = ["SIMULATOR.SCENE", "data/scene_datasets/gibson/Ackermanville.glb",
                    "DATASET.TYPE", "SpawnedObjectNav-v0",
                    "DATASET.DATA_PATH", PATH,
                    "DATASET.SPLIT", "test",
                    "TASK.TYPE", "SpawnedObjectNav-v0"]
    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml", EXTRA_CONFIG)
    cfg.TASK.defrost()
    cfg.TASK.SENSORS.extend(("OBJECTGOAL_CATEGORY", "OBJECTGOAL_APPEARANCE"))
    cfg.TASK.OBJECTGOAL_CATEGORY = habitat.Config()
    cfg.TASK.OBJECTGOAL_CATEGORY.TYPE = "SpawnedObjectGoalCategorySensor"
    cfg.TASK.OBJECTGOAL_APPEARANCE = habitat.Config()
    cfg.TASK.OBJECTGOAL_APPEARANCE.TYPE = "SpawnedObjectGoalAppearanceSensor"
    cfg.TASK.OBJECTGOAL_APPEARANCE.OUT_OF_CONTEXT = False
    cfg.TASK.OBJECTGOAL_APPEARANCE.OUT_OF_CONTEXT_POS = [0.0, 50.0, 0.0]
    cfg.TASK.OBJECTGOAL_APPEARANCE.MAX_VIEW_DISTANCE = 2.0
    cfg.TASK.OBJECTGOAL_APPEARANCE.MIN_VIEW_DISTANCE = 0.5
    cfg.TASK.OBJECTGOAL_APPEARANCE.ISLAND_RADIUS = 0.2
    cfg.TASK.OBJECTGOAL_APPEARANCE.RANDOM_OBJECT_ROTATION = "DISABLE"
    cfg.TASK.OBJECTGOAL_APPEARANCE.NUM_VIEWS = 5
    cfg.TASK.freeze()

    dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
    sim = habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    task = habitat.tasks.make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)

    assert sorted(task.sensor_suite.sensors.keys()) == ["objectgoal_appearance",
                                                        "objectgoal_category",
                                                        "pointgoal_with_gps_compass"]
    for episode in dataset.episodes:
        cfg.SIMULATOR.defrost()
        cfg.SIMULATOR.SCENE = episode.scene_id
        cfg.SIMULATOR.freeze()
        sim.reconfigure(cfg.SIMULATOR)

        obs = task.reset(episode)
        assert "objectgoal_category" in obs
        assert "objectgoal_appearance" in obs
