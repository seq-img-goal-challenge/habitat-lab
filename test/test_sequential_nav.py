import pytest
import numpy as np


def test_registration():
    from habitat.core.registry import registry
    from habitat.tasks.sequential_nav.sequential_nav import SequentialNavigationTask, \
                                                            FoundAction, \
                                                            SequentialPointGoalSensor, \
                                                            SequentialOnlinePointGoalSensor, \
                                                            SequentialTopDownMap, \
                                                            DistanceToNextGoal, \
                                                            SequentialSuccess, \
                                                            SequentialSPL, \
                                                            Progress, \
                                                            PPL

    assert registry.get_task("SequentialNav-v0") is SequentialNavigationTask
    assert registry.get_task_action("FoundAction") is FoundAction
    assert registry.get_sensor("SequentialPointGoalSensor") is SequentialPointGoalSensor
    assert registry.get_sensor("SequentialOnlinePointGoalSensor") \
            is SequentialOnlinePointGoalSensor
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
    from habitat.tasks.sequential_nav.sequential_nav import SequentialEpisode, SequentialStep
    from habitat.tasks.nav.nav import NavigationGoal
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config(opts=["TASK.TYPE", "SequentialNav-v0",
                           "TASK.POSSIBLE_ACTIONS", "['FOUND', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']"])
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


def test_sequential_pointgoal():
    from habitat.core.registry import registry
    from habitat.config.default import get_config
    from habitat.sims import make_sim
    from habitat.tasks import make_task
    from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    cfg = get_config(opts=["TASK.TYPE", "SequentialNav-v0",
                           "TASK.POSSIBLE_ACTIONS", "['FOUND', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']",
                           "TASK.SENSORS", "['SEQUENTIAL_POINTGOAL_SENSOR']"])
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
        task.reset(episode)
        follower = ShortestPathFollower(sim, cfg.TASK.SUCCESS_DISTANCE, False, False)
        while episode._current_step_index != 2:
            goal_pos = np.array(episode.steps[episode._current_step_index].goals[0].position)
            a = follower.get_next_action(goal_pos)
            obs = task.step({"action": a}, episode)
        assert obs["pointgoal"].shape == (episode.num_steps, 3)


def test_sequential_online_pointgoal():
    pass


def test_sequential_top_down_map():
    pass


def test_distance_to_next_goal():
    pass


def test_sequential_success():
    pass


def test_sequential_spl():
    pass


def test_progress():
    pass


def test_ppl():
    pass
