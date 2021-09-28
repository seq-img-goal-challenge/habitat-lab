import pytest
import numpy as np


def test_registration():
    from habitat.core.registry import registry
    from habitat.tasks.sequential_nav.sequential_nav import (SequentialNavigationTask,
                                                             FoundAction,
                                                             SequentialPointGoalSensor,
                                                             SequentialOnlinePointGoalSensor,
                                                             SequentialMapSensor,
                                                             SequentialEgoMapSensor,
                                                             SequentialOnlinePointGoalSensor,
                                                             SequentialTopDownMap,
                                                             DistanceToNextGoal,
                                                             SequentialSuccess,
                                                             SequentialSPL,
                                                             Progress,
                                                             PPL)

    assert registry.get_task("SequentialNav-v0") is SequentialNavigationTask
    assert registry.get_task_action("FoundAction") is FoundAction
    assert registry.get_sensor("SequentialPointGoalSensor") is SequentialPointGoalSensor
    assert registry.get_sensor("SequentialOnlinePointGoalSensor") \
            is SequentialOnlinePointGoalSensor
    assert registry.get_sensor("SequentialMapSensor") is SequentialMapSensor
    assert registry.get_sensor("SequentialEgoMapSensor") is SequentialEgoMapSensor
    assert registry.get_measure("SequentialTopDownMap") is SequentialTopDownMap
    assert registry.get_measure("DistanceToNextGoal") is DistanceToNextGoal
    assert registry.get_measure("SequentialSuccess") is SequentialSuccess
    assert registry.get_measure("SequentialSPL") is SequentialSPL
    assert registry.get_measure("Progress") is Progress
    assert registry.get_measure("PPL") is PPL


def test_default_config():
    from habitat.config.default import get_config

    cfg = get_config()

    assert "FOUND" in cfg.TASK.ACTIONS
    assert cfg.TASK.ACTIONS.FOUND.TYPE == "FoundAction"

    assert "SEQUENTIAL_POINTGOAL_SENSOR" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.TYPE == "SequentialPointGoalSensor"
    assert all(key in cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR
               for key in cfg.TASK.POINTGOAL_SENSOR)
    assert "SEQUENTIAL_MODE" in cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR
    assert "PADDING_VALUE" in cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR

    assert "SEQUENTIAL_ONLINE_POINTGOAL_SENSOR" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR.TYPE \
            == "SequentialOnlinePointGoalSensor"
    assert all(key in cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR
               for key in cfg.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR)
    assert "SEQUENTIAL_MODE" in cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR
    assert "PADDING_VALUE" in cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR

    assert "SEQUENTIAL_MAP_SENSOR" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_MAP_SENSOR.TYPE == "SequentialMapSensor"

    assert "SEQUENTIAL_EGO_MAP_SENSOR" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_EGO_MAP_SENSOR.TYPE == "SequentialEgoMapSensor"

    assert "SEQUENTIAL_TOP_DOWN_MAP" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_TOP_DOWN_MAP.TYPE == "SequentialTopDownMap"
    assert all(key in cfg.TASK.SEQUENTIAL_TOP_DOWN_MAP
               for key in cfg.TASK.TOP_DOWN_MAP)

    assert "DISTANCE_TO_NEXT_GOAL" in cfg.TASK
    assert cfg.TASK.DISTANCE_TO_NEXT_GOAL.TYPE == "DistanceToNextGoal"

    assert "SEQUENTIAL_SUCCESS" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_SUCCESS.TYPE == "SequentialSuccess"
    assert "SEQUENTIAL_SPL" in cfg.TASK
    assert cfg.TASK.SEQUENTIAL_SPL.TYPE == "SequentialSPL"
    assert "PROGRESS" in cfg.TASK
    assert cfg.TASK.PROGRESS.TYPE == "Progress"
    assert "PPL" in cfg.TASK
    assert cfg.TASK.PPL.TYPE == "PPL"


def _sample_reachable_point(sim, from_pos):
    d = np.inf
    while not np.isfinite(d):
        pos = sim.sample_navigable_point()
        d = sim.geodesic_distance(from_pos, pos)
    return pos


def _make_test_episode(sim):
    from habitat.tasks.sequential_nav.sequential_nav import SequentialEpisode, SequentialStep
    from habitat.tasks.nav.nav import NavigationGoal

    start = sim.sample_navigable_point()
    positions = [_sample_reachable_point(sim, start) for _ in range(4)]
    return SequentialEpisode(episode_id="ep_test",
                             scene_id=sim._current_scene,
                             start_position=start,
                             start_rotation=[0, 0, 0, 1],
                             steps=[SequentialStep(goals=[NavigationGoal(position=pos)])
                                    for pos in positions])


def test_sequential_nav_task():
    from habitat.core.registry import registry
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(123456)
        episode = _make_test_episode(sim)
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim)

        # Expected progression in sequence when stop is called at each goal
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        t = 0
        while task.is_episode_active:
            assert episode._current_step_index == t
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position,
                                dtype=np.float32)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
            if a == 0:
                t += 1
        assert t == episode.num_steps
        assert episode._current_step_index == episode.num_steps

        # Early termination when stop is called and current goal is out of reach
        task.reset(episode)
        assert episode._current_step_index == 0
        task.step({"action": 0}, episode)
        assert episode._current_step_index == -1
        assert not task.is_episode_active

        # Early termination when stop is called and a goal is in reach but not the correct one
        task.reset(episode)
        while task.is_episode_active:
            goal_pos = np.array(episode.steps[2].goals[0].position,
                                dtype=np.float32)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
        assert episode._current_step_index == -1


def test_gps_and_compass():
    import quaternion

    from habitat.core.registry import registry
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.SENSORS = ["GPS_SENSOR", "COMPASS_SENSOR"]
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(789123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        obs = task.reset(episode)
        assert all(key in obs for key in ("gps", "compass"))
        assert isinstance(obs["gps"], np.ndarray)
        assert obs["gps"].shape == (2,)
        assert obs["compass"].shape == (1,)
        phi, = obs["compass"]
        assert 0 <= phi <= 2*np.pi

        while task.is_episode_active:
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
            rot = np.quaternion(episode.start_rotation[3], *episode.start_rotation[:3])
            s = sim.get_agent_state()
            vec = s.position - episode.start_position
            x, _, z = (rot.inverse() * np.quaternion(0, *vec) * rot).vec
            assert np.allclose(obs["gps"], [-z, x], rtol=0.0001)
            r = rot.inverse() * s.rotation
            phi = 2 * np.arctan(r.y / r.w) if r.w != 0 else np.pi
            assert np.isclose(obs["compass"], phi)


def test_sequential_pointgoal():
    from habitat.core.registry import registry
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.SENSORS = ["SEQUENTIAL_POINTGOAL_SENSOR"]
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(123456)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        obs = task.reset(episode)
        assert "pointgoal" in obs
        assert isinstance(obs["pointgoal"], np.ndarray)
        assert obs["pointgoal"].shape == (episode.num_steps, 2)
        for (r, phi), step in zip(obs["pointgoal"], episode.steps):
            vec = np.array(step.goals[0].position) - episode.start_position
            assert np.allclose(r, np.linalg.norm(vec))
            assert np.allclose(phi, np.arctan2(-vec[0], -vec[2]))

        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.defrost()
        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.SEQUENTIAL_MODE = "MYOPIC"
        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.DIMENSIONALITY = 3
        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.freeze()
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        obs = task.reset(episode)
        assert obs["pointgoal"].shape == (3,)
        vec = np.array(episode.steps[0].goals[0].position) - episode.start_position
        assert np.allclose(obs["pointgoal"], vec)

        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.defrost()
        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.SEQUENTIAL_MODE = "SUFFIX"
        cfg.TASK.SEQUENTIAL_POINTGOAL_SENSOR.freeze()
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        while episode._current_step_index != 2:
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
        assert obs["pointgoal"].shape == (episode.num_steps, 3)
        for o, step in zip(obs["pointgoal"], episode.steps[episode._current_step_index:]):
            vec = np.array(step.goals[0].position) - episode.start_position
            assert np.allclose(o, vec)
        pad_len = episode.num_steps - episode._current_step_index
        assert np.all(obs["pointgoal"][-pad_len:] == 0)


def test_sequential_online_pointgoal():
    import quaternion

    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.SENSORS = ["SEQUENTIAL_ONLINE_POINTGOAL_SENSOR"]
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(456123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        for _ in range(6):
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
        assert "pointgoal_with_gps_compass" in obs
        assert isinstance(obs["pointgoal_with_gps_compass"], np.ndarray)
        assert obs["pointgoal_with_gps_compass"].shape == (episode.num_steps, 2)
        s = sim.get_agent_state()
        for (r, phi), step in zip(obs["pointgoal_with_gps_compass"], episode.steps):
            vec = step.goals[0].position - s.position
            vec = (s.rotation.inverse() * np.quaternion(0, *vec) * s.rotation).vec
            assert np.isclose(np.linalg.norm(vec), r)
            assert np.isclose(np.arctan2(-vec[0], -vec[2]), phi)

    cfg.defrost()
    cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR.SEQUENTIAL_MODE = "SUFFIX"
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(46123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        while episode._current_step_index == 0:
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
        assert obs["pointgoal_with_gps_compass"].shape == (episode.num_steps, 2)
        s = sim.get_agent_state()
        for (r, phi), step in zip(obs["pointgoal_with_gps_compass"], episode.steps[1:]):
            vec = step.goals[0].position - s.position
            vec = (s.rotation.inverse() * np.quaternion(0, *vec) * s.rotation).vec
            assert np.isclose(np.linalg.norm(vec), r)
            assert np.isclose(np.arctan2(-vec[0], -vec[2]), phi)
        assert np.all(obs["pointgoal_with_gps_compass"][-1] \
                      == cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR.PADDING_VALUE)

    cfg.defrost()
    cfg.TASK.SEQUENTIAL_ONLINE_POINTGOAL_SENSOR.SEQUENTIAL_MODE = "MYOPIC"
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(4563)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        while episode._current_step_index == 0:
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
        assert obs["pointgoal_with_gps_compass"].shape == (2,)
        s = sim.get_agent_state()
        r, phi = obs["pointgoal_with_gps_compass"]
        step = episode.steps[1]
        vec = step.goals[0].position - s.position
        vec = (s.rotation.inverse() * np.quaternion(0, *vec) * s.rotation).vec
        assert np.isclose(np.linalg.norm(vec), r)
        assert np.isclose(np.arctan2(-vec[0], -vec[2]), phi)


def test_map_sensor():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.utils.visualizations.maps import (MAP_TARGET_POINT_INDICATOR,
                                                   MAP_SOURCE_POINT_INDICATOR)
    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.SENSORS = ["SEQUENTIAL_MAP_SENSOR"]
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(456123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        obs = task.reset(episode)
        assert "map" in obs
        topdown_map = obs["map"]
        assert isinstance(topdown_map, np.ndarray)
        assert topdown_map.dtype == np.uint8
        res = cfg.TASK.SEQUENTIAL_MAP_SENSOR.RESOLUTION
        assert topdown_map.shape == (res, res)
        for a in (1, 2, 3, 1, 1):
            obs = task.step({"action": a}, episode)
            topdown_map = obs["map"]
            #TODO(gbono) check target and source (ie agent) indicator


def test_ego_map_sensor():
    import quaternion
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat.utils.visualizations.maps import (MAP_INVALID_POINT,
                                                   MAP_TARGET_POINT_INDICATOR,
                                                   MAP_SOURCE_POINT_INDICATOR)
    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.SENSORS = ["SEQUENTIAL_EGO_MAP_SENSOR"]
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(456123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        obs = task.reset(episode)
        assert "ego_map" in obs
        ego_map = obs["ego_map"]
        assert isinstance(ego_map, np.ndarray)
        assert ego_map.dtype == np.uint8
        mppx = cfg.TASK.SEQUENTIAL_EGO_MAP_SENSOR.METERS_PER_PIXEL
        res = int(2 * cfg.TASK.SEQUENTIAL_EGO_MAP_SENSOR.VISIBILITY / mppx)
        while task.is_episode_active:
            goal_pos = episode.steps[episode._current_step_index].goals[0].position
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
            ego_map = obs["ego_map"]
            assert ego_map.shape == (res, res)
            s = sim.get_agent_state()
            for step in episode.steps:
                rel_pos = step.goals[0].position - s.position
                rel_pos = (s.rotation.conj() * np.quaternion(0, *rel_pos) * s.rotation).vec
                j, _, i = (rel_pos / mppx).astype(np.int64) + res // 2
                if 0 <= i < res and 0 <= j < res:
                    assert ego_map[i, j] in (MAP_INVALID_POINT, MAP_TARGET_POINT_INDICATOR)


def test_sequential_top_down_map():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.utils.visualizations.maps import to_grid, MAP_TARGET_POINT_INDICATOR

    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.MEASUREMENTS = ["SEQUENTIAL_TOP_DOWN_MAP"]
    cfg.TASK.SEQUENTIAL_TOP_DOWN_MAP.DRAW_SHORTEST_PATH = False
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(456123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        task.reset(episode)
        task.measurements.reset_measures(episode=episode, task=task)
        for a in (1, 2 ,2):
            task.step({"action": a}, episode)
            task.measurements.update_measures(episode=episode, action={"action": a}, task=task)
        m = task.measurements.get_metrics()
        assert "top_down_map" in m
        assert isinstance(m["top_down_map"], dict)
        assert all(key in m["top_down_map"] for key in ("map", "fog_of_war_mask",
                                                        "agent_map_coord", "agent_angle"))
        topdown_map = m["top_down_map"]["map"]
        assert isinstance(topdown_map, np.ndarray)
        mask = m["top_down_map"]["fog_of_war_mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == topdown_map.shape
        ij = m["top_down_map"]["agent_map_coord"]
        assert isinstance(ij, tuple)
        assert len(ij) == 2
        assert all(isinstance(x, int) for x in ij)
        assert all(0 <= x < s for x, s in zip(ij, topdown_map.shape))
        phi = m["top_down_map"]["agent_angle"]
        assert isinstance(phi, float)
        assert 0 <= phi <= 2 * np.pi

        for step in episode.steps:
            x, _, z = step.goals[0].position
            ij = to_grid(z, x, topdown_map.shape, sim)
            assert topdown_map[ij] == MAP_TARGET_POINT_INDICATOR


def test_distance_to_next_goal():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.TASK.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_NEXT_GOAL"]
    cfg.TASK.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(456123)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        task.measurements.reset_measures(episode=episode, task=task)
        m = task.measurements.get_metrics()
        assert "distance_to_next_goal" in m
        assert isinstance(m["distance_to_next_goal"], float)
        while task.is_episode_active:
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            d = sim.geodesic_distance(sim.get_agent_state().position, goal_pos)
            assert np.isclose(m["distance_to_next_goal"], d)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
            task.measurements.update_measures(episode=episode, action={"action": a}, task=task)
            m = task.measurements.get_metrics()


def test_success():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_NEXT_GOAL", "SEQUENTIAL_SUCCESS"]
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(789456)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        task.measurements.reset_measures(episode=episode, task=task)
        m = task.measurements.get_metrics()
        assert "seq_success" in m
        assert isinstance(m["seq_success"], float)
        while task.is_episode_active:
            assert m["seq_success"] == 0.0
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
            task.measurements.update_measures(episode=episode, action={"action": a}, task=task)
            m = task.measurements.get_metrics()
        assert m["seq_success"] == 1.0


def test_spl():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.MEASUREMENTS = ["DISTANCE_TO_NEXT_GOAL", "SEQUENTIAL_SUCCESS", "SEQUENTIAL_SPL"]
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(789456)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        task.measurements.reset_measures(episode=episode, task=task)
        m = task.measurements.get_metrics()
        assert "seq_spl" in m
        assert isinstance(m["seq_spl"], float)
        while task.is_episode_active:
            assert m["seq_spl"] == 0.0
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
            task.measurements.update_measures(episode=episode, action={"action": a}, task=task)
            m = task.measurements.get_metrics()
        assert 0.0 < m["seq_spl"] <= 1.0


def test_progress():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.MEASUREMENTS = ["PROGRESS"]
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(15973)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        task.measurements.reset_measures(episode=episode, task=task)
        m = task.measurements.get_metrics()
        assert "progress" in m
        assert isinstance(m["progress"], float)
        while task.is_episode_active:
            assert m["progress"] == episode._current_step_index / episode.num_steps
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
            task.measurements.update_measures(episode=episode, action={"action": a}, task=task)
            m = task.measurements.get_metrics()
        assert m["progress"] == 1.0


def test_ppl():
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config()
    cfg.defrost()
    cfg.TASK.TYPE = "SequentialNav-v0"
    cfg.TASK.POSSIBLE_ACTIONS = ["FOUND", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    cfg.TASK.MEASUREMENTS = ["PROGRESS", "PPL"]
    cfg.freeze()
    with make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
        sim.seed(15973)
        episode = _make_test_episode(sim)
        dataset = SequentialDataset()
        dataset.episodes = [episode]
        task = make_task(cfg.TASK.TYPE, config=cfg.TASK, sim=sim, dataset=dataset)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        task.reset(episode)
        task.measurements.reset_measures(episode=episode, task=task)
        m = task.measurements.get_metrics()
        assert "ppl" in m
        assert isinstance(m["ppl"], float)
        while task.is_episode_active:
            assert 0.0 <= m["ppl"] <= episode._current_step_index / episode.num_steps
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            task.step({"action": a}, episode)
            task.measurements.update_measures(episode=episode, action={"action": a}, task=task)
            m = task.measurements.get_metrics()
        assert 0.0 < m["ppl"] <= 1.0
