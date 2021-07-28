import os.path
import copy

import numpy as np

import pytest


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


def test_spawn_pos_distrib_init():
    import habitat
    from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        distrib = SpawnPositionDistribution(sim)

        assert distrib._origin is not None
        assert isinstance(distrib._origin, np.ndarray)
        assert distrib._origin.shape == (3,)
        assert distrib._origin.dtype == np.float32
        assert distrib._origin[1] == distrib._height

        assert distrib._nav_mask is not None
        assert isinstance(distrib._nav_mask, np.ndarray)
        assert distrib._nav_mask.ndim == 2
        assert distrib._nav_mask.dtype == np.bool

        assert distrib._edges is not None
        assert isinstance(distrib._edges, np.ndarray)
        assert distrib._edges.shape == distrib._nav_mask[:-1, :-1].shape
        assert distrib._edges.dtype == np.float32
        assert np.all((distrib._edges == 0) | (distrib._edges == 1))

        assert distrib._distrib is not None
        assert isinstance(distrib._distrib, np.ndarray)
        assert distrib._distrib.shape == distrib._edges.shape
        assert distrib._distrib.dtype == np.float32
        assert np.all((distrib._distrib >= 0) & (distrib._distrib < 1))

        assert distrib._cumul is not None
        assert isinstance(distrib._cumul, np.ndarray)
        assert distrib._cumul.shape == (distrib._distrib.size,)
        assert distrib._cumul.dtype == np.float32

        assert distrib._conn_comp_masks is None
        assert distrib._conn_comp_weights is None
        assert distrib._num_conn_comp == 0


def test_spawn_pos_distrib_update():
    import habitat
    from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        distrib = SpawnPositionDistribution(sim)

        prv_origin = distrib._origin
        prv_nav_mask = distrib._nav_mask
        prv_edges = distrib._edges
        prv_distrib = distrib._distrib
        prv_cumul = distrib._cumul

        distrib.inflate_radius = 3.0
        assert distrib._origin is prv_origin
        assert distrib._nav_mask is prv_nav_mask
        assert distrib._edges is prv_edges
        assert distrib._distrib is not prv_distrib
        assert distrib._cumul is not prv_cumul
        prv_distrib = distrib._distrib
        prv_cumul = distrib._cumul

        distrib.resolution = 0.5
        assert distrib._origin is prv_origin
        assert distrib._nav_mask is not prv_nav_mask
        assert distrib._edges is not prv_edges
        assert distrib._distrib is not prv_distrib
        assert distrib._cumul is not prv_cumul
        prv_edges = distrib._edges
        prv_distrib = distrib._distrib
        prv_cumul = distrib._cumul

        distrib.margin = 0.2
        assert distrib._origin is not prv_origin
        assert distrib._nav_mask is not prv_nav_mask
        assert distrib._edges is not prv_edges
        assert distrib._distrib is not prv_distrib
        assert distrib._cumul is not prv_cumul


def test_spawn_pos_distrib_conn_comp():
    import habitat
    from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        distrib = SpawnPositionDistribution(sim)
        n_cc = distrib.get_num_connected_components()

        assert distrib._conn_comp_masks is not None
        assert isinstance(distrib._conn_comp_masks, np.ndarray)
        assert distrib._conn_comp_masks.shape == distrib._distrib.shape + (n_cc,)
        assert distrib._conn_comp_masks.dtype == np.bool
        assert np.all(distrib._conn_comp_masks.any(axis=-1) == distrib._nav_mask[:-1, :-1])

        assert distrib._conn_comp_weights is not None
        assert isinstance(distrib._conn_comp_masks, np.ndarray)
        assert distrib._conn_comp_weights.shape == (n_cc,)
        assert distrib._conn_comp_weights.dtype == np.float32

        assert distrib._num_conn_comp > 0
        assert distrib._num_conn_comp == n_cc


def test_spawn_pos_distrib_getters():
    import habitat
    from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        distrib = SpawnPositionDistribution(sim)

        o = distrib.get_origin()
        assert np.all(o == distrib._origin)
        assert o is not distrib._origin

        edges = distrib.get_map_edges()
        assert np.all(edges == distrib._edges)
        assert edges is not distrib._edges

        spatial_distrib = distrib.get_spatial_distribution()
        assert spatial_distrib.shape == distrib._distrib.shape
        assert spatial_distrib.dtype == np.float32
        assert np.isclose(spatial_distrib.sum(), 1)
        assert spatial_distrib is not distrib._distrib

        n_cc = distrib.get_num_connected_components()
        assert n_cc == distrib._num_conn_comp

        cc = distrib.get_connected_component(0)
        assert isinstance(cc, np.ndarray)
        assert cc.shape == distrib._distrib.shape
        assert cc.dtype == np.bool
        assert np.all(cc == distrib._conn_comp_masks[:, :, 0])
        assert cc is not distrib._conn_comp_masks[:, :, 0]


def test_spawn_pos_distrib_coord_conv():
    import habitat
    from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        distrib = SpawnPositionDistribution(sim)

        lower, upper = sim.pathfinder.get_bounds()
        ij = distrib.world_to_map(lower)
        back = distrib.map_to_world(ij)
        assert isinstance(ij, np.ndarray)
        assert ij.shape == (2,)
        assert ij.dtype == np.int64
        assert np.all(ij == 0)
        assert np.allclose(back, np.array([lower[0], distrib.height, lower[2]]),
                           atol=distrib.resolution)

        ij = distrib.world_to_map(upper)
        back = distrib.map_to_world(ij)
        assert np.all(ij == distrib._nav_mask.shape)
        assert np.allclose(back, np.array([upper[0], distrib.height, upper[2]]),
                           atol=distrib.resolution)

        pos = sim.get_agent_state().position
        i, j = distrib.world_to_map(pos)
        back = distrib.map_to_world(i, j)
        assert distrib._nav_mask[i, j]
        assert np.allclose(back, pos, atol=distrib.resolution)


def test_spawn_pos_distrib_sample():
    import habitat
    from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        distrib = SpawnPositionDistribution(sim)
        distrib.seed(123456)
        lower, upper = sim.pathfinder.get_bounds()

        positions = distrib.sample(50)
        i, j = distrib.world_to_map(positions).T
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (50, 3)
        assert positions.dtype == np.float32
        assert np.allclose(positions[:, 1], distrib.height)
        assert np.all((positions >= lower) & (positions <= upper))
        assert np.all(distrib._nav_mask[i, j])

        positions = distrib.sample_from_connected_component(100, 0)
        i, j = distrib.world_to_map(positions).T
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (100, 3)
        assert positions.dtype == np.float32
        assert np.allclose(positions[:, 1], distrib.height)
        assert np.all((positions >= lower) & (positions <= upper))
        assert np.all(distrib._nav_mask[i, j])
        assert np.all(distrib._conn_comp_masks[i, j, 0])
        assert not np.any(distrib._conn_comp_masks[i, j, 1:])

        start_pos = sim.sample_navigable_point()
        positions = distrib.sample_reachable_from_position(10, start_pos)
        i, j = distrib.world_to_map(positions).T
        _, cc_indices = np.nonzero(distrib._conn_comp_masks[i, j])
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (10, 3)
        assert positions.dtype == np.float32
        assert np.allclose(positions[:, 1], distrib.height)
        assert np.all((positions >= lower) & (positions <= upper))
        assert np.all(distrib._nav_mask[i, j])
        assert np.all(cc_indices == cc_indices[0])


def test_create_object_pool():
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_object_pool

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"

    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    pool = create_object_pool(OBJECTS_DIR)
    seen_idx = set()
    assert isinstance(pool, list)
    for name, idx, templates in pool:
        assert isinstance(name, str)
        assert name in {"box", "chair", "donut", "sphere"}
        assert isinstance(idx, int)
        assert idx not in seen_idx
        seen_idx.add(idx)
        assert isinstance(templates, list)
        assert 0 < len(templates) < 4
        for tmpl in templates:
            assert isinstance(tmpl, str)
            assert tmpl.endswith(OBJECT_EXT)
            assert os.path.isfile(tmpl)


def test_create_scene_pool():
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_scene_pool

    SCENES_DIR = "data/scene_datasets/habitat-test-scenes"
    SCENE_EXT = ".glb"

    if not os.path.isdir(SCENES_DIR):
        pytest.skip(f"Test scenes '{SCENES_DIR}' not available.")

    pool = create_scene_pool(SCENES_DIR)
    assert isinstance(pool, list)
    for scene in pool:
        assert isinstance(scene, str)
        assert scene.endswith(SCENE_EXT)
        assert os.path.isfile(scene)


def test_spawn_objects_fixed_rot():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import spawn_objects
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal
    from habitat_sim.physics import MotionType

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)
        template_ids = [os.path.join(OBJECTS_DIR, tmpl, tmpl + OBJECT_EXT)
                        for tmpl in ("chair", "donut")]
        positions = np.array([sim.sample_navigable_point() for _ in template_ids])
        rng = np.random.default_rng(123456)

        goals = spawn_objects(sim, template_ids, positions, rng=rng)
        assert isinstance(goals, list)
        for goal, tmpl_id, pos in zip(goals, template_ids, positions):
            assert isinstance(goal, SpawnedObjectGoal)
            assert np.allclose(goal.position, pos)
            assert goal.rotation == [0, 0, 0, 1]
            assert goal.object_template_id == tmpl_id
            assert goal.view_points == []
            assert goal.radius is None
            assert goal._spawned_object_id is not None
            assert goal._appearance_cache is None

            assert sim.get_object_motion_type(goal._spawned_object_id) == MotionType.STATIC

            node = sim.get_object_scene_node(goal._spawned_object_id)
            assert np.allclose(node.translation, pos)
            assert np.allclose([*node.rotation.vector, node.rotation.scalar], [0, 0, 0, 1])


def test_spawn_objects_vertical_rot():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import spawn_objects, ObjectRotation

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)
        template_ids = [os.path.join(OBJECTS_DIR, tmpl, tmpl + OBJECT_EXT)
                        for tmpl in ("chair", "donut")]
        positions = np.array([sim.sample_navigable_point() for _ in template_ids])
        rng = np.random.default_rng(123456)

        goals = spawn_objects(sim, template_ids, positions, ObjectRotation.VERTICAL, rng=rng)
        for goal in goals:
            assert np.isclose(goal.rotation[0], 0)
            assert np.isclose(goal.rotation[2], 0)
            assert np.isclose(goal.rotation[1]**2 + goal.rotation[3]**2, 1)

            node = sim.get_object_scene_node(goal._spawned_object_id)
            assert np.allclose(node.translation, goal.position)
            assert np.allclose([*node.rotation.vector, node.rotation.scalar], goal.rotation)


def test_spawn_objects_free_rot():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import spawn_objects, ObjectRotation

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)
        template_ids = [os.path.join(OBJECTS_DIR, tmpl, tmpl + OBJECT_EXT)
                        for tmpl in ("chair", "donut")]
        positions = np.array([sim.sample_navigable_point() for _ in template_ids])
        rng = np.random.default_rng(123456)

        goals = spawn_objects(sim, template_ids, positions, ObjectRotation.FREE, rng=rng)
        for goal in goals:
            assert np.isclose(np.linalg.norm(goal.rotation), 1)

            node = sim.get_object_scene_node(goal._spawned_object_id)
            assert np.allclose(node.translation, goal.position)
            assert np.allclose([*node.rotation.vector, node.rotation.scalar], goal.rotation)


def test_recompute_navmesh():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import spawn_objects, recompute_navmesh_with_static_objects

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"

    sim_cfg = habitat.get_config().SIMULATOR
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)
        template_ids = [os.path.join(OBJECTS_DIR, tmpl, tmpl + OBJECT_EXT)
                        for tmpl in ("chair", "donut")]
        positions = np.array([sim.sample_navigable_point() for _ in template_ids])
        rng = np.random.default_rng(123456)

        goals = spawn_objects(sim, template_ids, positions, rng=rng)
        recompute_navmesh_with_static_objects(sim)


def test_find_view_points():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import spawn_objects, recompute_navmesh_with_static_objects, find_view_points

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"
    NUM_RADII = 5
    NUM_ANGLES = 12

    sim_cfg = habitat.get_config().SIMULATOR
    sim_cfg.defrost()
    sim_cfg.AGENT_0.SENSORS = ['DEPTH_SENSOR']
    sim_cfg.freeze()
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)
        template_ids = [os.path.join(OBJECTS_DIR, tmpl, tmpl + OBJECT_EXT)
                        for tmpl in ("chair", "donut")]
        positions = np.array([sim.sample_navigable_point() for _ in template_ids])
        rng = np.random.default_rng(123456)

        src_goals = spawn_objects(sim, template_ids, positions, rng=rng)
        cp_goals = copy.deepcopy(src_goals)
        recompute_navmesh_with_static_objects(sim)
        start_pos = sim.get_agent_state().position

        goals = find_view_points(sim, src_goals, start_pos,
                                 num_radii=NUM_RADII, num_angles=NUM_ANGLES)
        sensor_pos = np.array(sim.habitat_config.DEPTH_SENSOR.POSITION)
        assert isinstance(goals, list)
        for goal, src_goal, cp_goal in zip(goals, src_goals, cp_goals):
            assert goal is src_goal
            assert goal is not cp_goal
            assert np.allclose(goal.position, cp_goal.position)
            assert np.allclose(goal.rotation, cp_goal.rotation)
            assert goal.object_template_id == cp_goal.object_template_id
            assert goal.radius is None
            assert goal._appearance_cache is None
            assert 0 < len(goal.view_points) <= NUM_RADII * NUM_ANGLES
            for view_pt in goal.view_points:
                ag_pos = view_pt.position - sensor_pos
                assert sim.pathfinder.is_navigable(ag_pos)
                assert np.isfinite(sim.geodesic_distance(start_pos, ag_pos))
                assert view_pt.iou > 0


def test_generate_spawned_objectgoals():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import generate_spawned_objectgoals, ObjectRotation
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal

    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"

    sim_cfg = habitat.get_config().SIMULATOR
    sim_cfg.defrost()
    sim_cfg.AGENT_0.SENSORS = ['DEPTH_SENSOR']
    sim_cfg.freeze()
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)
        start_pos = sim.sample_navigable_point()
        template_ids = [os.path.join(OBJECTS_DIR, tmpl, tmpl + OBJECT_EXT)
                        for tmpl in ("chair", "donut")]
        rng = np.random.default_rng(123456)

        goals = generate_spawned_objectgoals(sim, start_pos, template_ids,
                                             ObjectRotation.VERTICAL, rng)
        sensor_pos = np.array(sim.habitat_config.DEPTH_SENSOR.POSITION)
        assert isinstance(goals, list)
        for goal, tmpl_id in zip(goals, template_ids):
            assert isinstance(goal, SpawnedObjectGoal)
            assert goal.object_template_id == tmpl_id
            assert goal.radius is None
            assert goal._spawned_object_id is not None
            assert goal._appearance_cache is None
            assert 0 < len(goal.view_points) <= 60
            for view_pt in goal.view_points:
                ag_pos = view_pt.position - sensor_pos
                assert sim.pathfinder.is_navigable(ag_pos)
                assert np.isfinite(sim.geodesic_distance(start_pos, ag_pos))
                assert view_pt.iou > 0


def test_generate_episode():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_object_pool, generate_spawned_objectnav_episode, ObjectRotation
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectNavEpisode

    OBJECTS_DIR = "data/object_datasets/test_objects"
    MAX_GOALS = 2
    NUM_RETRIES = 2

    sim_cfg = habitat.get_config().SIMULATOR
    sim_cfg.defrost()
    sim_cfg.AGENT_0.SENSORS = ['DEPTH_SENSOR']
    sim_cfg.freeze()
    if not os.path.isfile(sim_cfg.SCENE):
        pytest.skip(f"Test scene '{sim_cfg.SCENE}' not available.")
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    with habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg) as sim:
        sim.seed(123456)

        obj_pool = create_object_pool(OBJECTS_DIR)
        rng = np.random.default_rng(123456)
        episode = generate_spawned_objectnav_episode(sim, "ep0", MAX_GOALS, obj_pool,
                                                     ObjectRotation.FIXED, NUM_RETRIES, rng)
        lower, upper = sim.pathfinder.get_bounds()
        assert isinstance(episode, SpawnedObjectNavEpisode)
        assert episode.episode_id == "ep0"
        assert episode.scene_id == sim.habitat_config.SCENE
        assert np.all((episode.start_position >= lower) & (episode.start_position <= upper))
        assert sim.pathfinder.is_navigable(episode.start_position)
        assert np.isclose(episode.start_rotation[0], 0)
        assert np.isclose(episode.start_rotation[2], 0)
        assert np.isclose(episode.start_rotation[1]**2 + episode.start_rotation[3]**2, 1)
        assert episode.object_category in {name for name, _, _ in obj_pool}
        assert episode.object_category_index == next(idx for name, idx, _ in obj_pool \
                                                     if name == episode.object_category)
        assert 0 < len(episode.goals) <= MAX_GOALS


def test_generate_dataset():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset \
            import SpawnedObjectNavDatasetV0
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import generate_spawned_objectnav_dataset, ObjectRotation, ExistBehavior
    from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode

    CFG_PATH = "configs/tasks/spawned_objectnav.yaml"
    NUM_EPISODES = 3
    MAX_GOALS = 2
    NUM_RETRIES = 2
    SEED = 123789
    SCENES_DIR = "data/scene_datasets/habitat-test-scenes"
    SCENE_EXT = ".glb"
    if not os.path.isdir(SCENES_DIR):
        pytest.skip(f"Test scenes '{SCENES_DIR}' not available.")
    OBJECTS_DIR = "data/object_datasets/test_objects"
    OBJECT_EXT = ".object_config.json"
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")

    generate_spawned_objectnav_dataset(CFG_PATH, [], SCENES_DIR, OBJECTS_DIR, 
                                       NUM_EPISODES, MAX_GOALS, ObjectRotation.FIXED,
                                       ExistBehavior.OVERRIDE, NUM_RETRIES, SEED)
    data_cfg = habitat.get_config(CFG_PATH).DATASET
    data_path = data_cfg.DATA_PATH.format(split=data_cfg.SPLIT)
    assert os.path.isfile(data_path)

    dataset = habitat.make_dataset(data_cfg.TYPE, config=data_cfg)
    assert isinstance(dataset, SpawnedObjectNavDatasetV0)
    assert len(dataset.episodes) == NUM_EPISODES
    seen_id = set()
    for episode in dataset.episodes:
        assert isinstance(episode, SpawnedObjectNavEpisode)
        assert episode.episode_id not in seen_id
        seen_id.add(episode.episode_id)
        assert episode.scene_id.endswith(SCENE_EXT)
        assert os.path.isfile(episode.scene_id)
        assert 0 < len(episode.goals) <= MAX_GOALS
        for goal in episode.goals:
            assert isinstance(goal, SpawnedObjectGoal)
            assert goal.object_template_id.endswith(OBJECT_EXT)
            assert os.path.isfile(goal.object_template_id)
            assert goal._spawned_object_id is None
            assert goal._appearance_cache is None
            assert 0 < len(goal.view_points) <= 60


def test_task():
    import habitat
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    CFG_PATH = "configs/tasks/spawned_objectnav.yaml"

    cfg = habitat.get_config(CFG_PATH)
    with habitat.Env(cfg) as env:
        obs = env.reset()
        assert 'rgb' in obs
        assert 'depth' in obs
        assert 'objectgoal_appearance' in obs
        assert 'objectgoal_category' in obs

        m = env.get_metrics()
        assert 'distance_to_goal' in m
        assert 'success' in m
        assert 'spl' in m

        follower = ShortestPathFollower(env.sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        for _ in env.episodes:
            obs = env.reset()
            prv_d = None
            while not env.episode_over:
                a = follower.get_next_action(env.current_episode.goals[0].position)
                obs = env.step(a)
                m = env.get_metrics()
                d = m['distance_to_goal']
                if prv_d is not None:
                    assert d <= prv_d
                prv_d = d
            assert m['distance_to_goal'] <= cfg.TASK.SUCCESS_DISTANCE
            assert m['success'] == 1.0
            assert m['spl'] > 0.5


def test_objectgoal_category_sensor():
    import habitat
    from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
            import create_object_pool

    OBJECTS_DIR = "data/object_datasets/test_objects"
    if not os.path.isdir(OBJECTS_DIR):
        pytest.skip(f"Test objects '{OBJECTS_DIR}' not available.")
    CFG_PATH = "configs/tasks/spawned_objectnav.yaml"
    ACTIONS = (1, 2, 3, 1)

    pool = create_object_pool(OBJECTS_DIR)
    cfg = habitat.get_config(CFG_PATH)
    with habitat.Env(cfg) as env:
        for _ in env.episodes:
            obs = env.reset()
            ep = env.current_episode
            assert 'objectgoal_category' in obs
            for a in ACTIONS:
                cat = obs['objectgoal_category']
                assert isinstance(cat, np.ndarray)
                assert cat.shape == (1,)
                assert cat.dtype == np.int64
                assert cat == ep.object_category_index
                cat_name = next(name for name, idx, _ in pool if idx == cat)
                assert cat_name == ep.object_category
                obs = env.step(a)


def test_objectgoal_appearance_sensor():
    import habitat

    CFG_PATH = "configs/tasks/spawned_objectnav.yaml"
    ACTIONS = (1, 2, 3, 1)

    cfg = habitat.get_config(CFG_PATH)
    num_views = cfg.TASK.SPAWNED_OBJECTGOAL_APPEARANCE.NUM_VIEWS
    width = cfg.SIMULATOR.RGB_SENSOR.WIDTH
    height = cfg.SIMULATOR.RGB_SENSOR.HEIGHT
    with habitat.Env(cfg) as env:
        for _ in env.episodes:
            obs = env.reset()
            ep = env.current_episode
            prv_views = None

            assert 'objectgoal_appearance' in obs
            for a in ACTIONS:
                views = obs['objectgoal_appearance']
                assert isinstance(views, np.ndarray)
                assert views.shape == (num_views, height, width, 3)
                assert views.dtype == np.uint8
                if prv_views is not None:
                    assert np.allclose(prv_views, views)
                prv_views = views.copy()
                obs = env.step(a)


    cfg.defrost()
    cfg.TASK.SPAWNED_OBJECTGOAL_APPEARANCE.OUT_OF_CONTEXT = True
    cfg.freeze()
    with habitat.Env(cfg) as env:
        for _ in env.episodes:
            obs = env.reset()
            ep = env.current_episode
            prv_views = None

            assert 'objectgoal_appearance' in obs
            for a in ACTIONS:
                views = obs['objectgoal_appearance']
                assert isinstance(views, np.ndarray)
                assert views.shape == (num_views, height, width, 3)
                assert views.dtype == np.uint8
                if prv_views is not None:
                    assert np.allclose(prv_views, views)
                prv_views = views.copy()
                obs = env.step(a)
