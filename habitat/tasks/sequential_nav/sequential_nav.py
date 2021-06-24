from typing import Any, Optional, List, Dict

import attr
import numpy as np
from gym import Space, spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, Action
from habitat.core.registry import registry
from habitat.core.simulator import Simulator, Sensor, SensorTypes, Observations
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, NavigationEpisode, \
                                  PointGoalSensor, IntegratedPointGoalGPSAndCompassSensor, \
                                  TopDownMap
from habitat.utils.visualizations import maps


@attr.s(auto_attribs=True, kw_only=True)
class SequentialStep:
    goals: List[NavigationGoal] = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class SequentialEpisode(Episode):
    steps: List[SequentialStep] = attr.ib(default=None, validator=not_none_validator)
    _current_step_index: int = attr.ib(init=False, default=0)

    @property
    def num_steps(self):
        return len(self.steps)

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._current_step_index = 0


class SequentialDataset(Dataset):
    episodes: List[SequentialEpisode]

    def get_max_sequence_len(self) -> int:
        return max(len(episode.steps) for episode in self.episodes)


@registry.register_task(name="SequentialNav-v0")
class SequentialNavigationTask(NavigationTask):
    is_stop_called: bool
    _success_d: float

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self.is_stop_called = False
        self._success_d = config.SUCCESS_DISTANCE
        super().__init__(*args, config=config, **kwargs)

    def reset(self, episode: SequentialEpisode) -> Observations:
        self.is_stop_called = False
        episode._current_step_index = 0
        return super().reset(episode)

    def _is_on_last_step(self, episode: SequentialEpisode) -> bool:
        return episode._current_step_index == len(episode.steps) - 1

    def _check_episode_is_active(self, *args: Any, episode: SequentialEpisode,
                                 **kwargs: Any) -> bool:
        if self.is_stop_called:
            if self._is_on_last_step(episode):
                return False
            else:
                ag_pos = self._sim.get_agent_state().position
                step = episode.steps[episode._current_step_index]
                d = self._sim.geodesic_distance(ag_pos, [goal.position for goal in step.goals])
                if d <= self._success_d or any(goal.radius is not None and d <= goal.radius
                                               for goal in step.goals):
                    episode._current_step_index += 1
                    self.is_stop_called = False
                    return True
                else:
                    return False
        else:
            return True


def make_sequential(base_sensor_cls, *, name=None):
    if name is None:
        name = "Sequential" + base_sensor_cls.__name__

    @registry.register_sensor(name=name)
    class SequentialSensor(base_sensor_cls):
        _max_seq_len: int
        _pad_val: int
        _seq_mode: str

        def __init__(self, *args: Any, sim: Simulator, config: Config,
                     dataset: SequentialDataset, **kwargs: Any) -> None:
            self._max_seq_len = dataset.get_max_sequence_len()
            self._pad_val = config.PADDING_VALUE
            self._seq_mode = config.SEQUENTIAL_MODE
            super().__init__(*args, sim=sim, config=config, dataset=dataset, **kwargs)

        def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
            src_space = super()._get_observation_space(*args, **kwargs)
            if self.config.SEQUENTIAL_MODE == "MYOPIC":
                return src_space
            elif isinstance(src_space, spaces.Box):
                extended_shape = (self._max_seq_len,) + src_space.shape
                low = np.broadcast_to(src_space.low, extended_shape)
                high = np.broadcast_to(src_space.high, extended_shape)
                return spaces.Box(low=np.minimum(low, self._pad_val),
                                  high=np.maximum(high, self._pad_val),
                                  dtype=src_space.dtype)
            else:
                raise NotImplementedError(f"Cannot make sequential sensor" \
                        + "for observation space of type '{type(src_space)}'")

        def _get_observation_for_step(self, episode: SequentialEpisode, step_index: int,
                                      *args: Any, **kwargs: Any) -> Any:
            step = episode.steps[step_index]
            step_id = f"{episode.episode_id}_{step_index}"
            step_as_episode = NavigationEpisode(episode_id=step_id,
                                                scene_id=episode.scene_id,
                                                start_position=episode.start_position,
                                                start_rotation=episode.start_rotation,
                                                goals=step.goals)
            return super().get_observation(*args, episode=step_as_episode, **kwargs)

        def get_observation(self, *args: Any, episode: SequentialEpisode,
                            **kwargs: Any) -> Any:
            if self._seq_mode == "MYOPIC":
                return self._get_observation_for_step(episode, episode._current_step_index,
                                                      *args, **kwargs)
            if self._seq_mode == "PREFIX":
                obs_seq = [self._get_observation_for_step(episode, idx, *args, **kwargs)
                           for idx in range(0, episode._current_step_index + 1)]
            elif self._seq_mode == "SUFFIX":
                obs_seq = [self._get_observation_for_step(episode, idx, *args, **kwargs)
                           for idx in range(episode._current_step_index, episode.num_steps)]
            elif self._seq_mode == "FULL":
                obs_seq = [self._get_observation_for_step(episode, idx, *args, **kwargs)
                           for idx in range(0, episode.num_steps)]
            obs_seq = np.stack(obs_seq, 0)
            pad_len = self._max_seq_len - obs_seq.shape[0]
            pad = np.full((pad_len,) + obs_seq.shape[1:], self._pad_val)
            return np.concatenate([obs_seq, pad], 0)

    return SequentialSensor


SequentialPointGoalSensor = make_sequential(PointGoalSensor)
SequentialOnlinePointGoalSensor = make_sequential(IntegratedPointGoalGPSAndCompassSensor,
                                                  name="SequentialOnlinePointGoalSensor")


@registry.register_measure
class SequentialTopDownMap(TopDownMap):
    def _compute_shortest_path(self, episode: SequentialEpisode,
                                     start_pos: List[float]) -> List[List[float]]:
        last = [(start_pos, 0.0, None)]
        values = []
        for step in episode.steps[episode._current_step_index:]:
            values.append(last)
            last = [(g.position, *min((d + self._sim.geodesic_distance(pos, g.position), i)
                                      for i, (pos, d, _) in enumerate(last)))
                    for g in step.goals]

        pos, _, back = min(last, key=lambda tup: tup[1])
        path = []
        while back is not None:
            prv_pos, _, back = values.pop()[back]
            path.extend(reversed(self._sim.get_straight_shortest_path_points(prv_pos, pos)))
            pos = prv_pos
        path.reverse()
        return path

    def _draw_goals_view_points(self, episode: SequentialEpisode) -> None:
        for step in episode.steps:
            super()._draw_goals_view_points(step)

    def _draw_goals_positions(self, episode: SequentialEpisode) -> None:
        for step in episode.steps:
            super()._draw_goals_positions(step)

    def _draw_goals_aabb(self, episode: SequentialEpisode) -> None:
        for step in episode.steps:
            super()._draw_goals_aabb(step)

    def _draw_shortest_path(self, episode: SequentialEpisode,
                                  agent_position: List[float]) -> None:
        if self._config.DRAW_SHORTEST_PATH:
            path = self._compute_shortest_path(episode, agent_position)
            self._shortest_path_points = [maps.to_grid(p[2], p[0],
                                                       self._top_down_map.shape[0:2],
                                                       sim=self._sim)
                                          for p in path]
            maps.draw_path(self._top_down_map, self._shortest_path_points,
                           maps.MAP_SHORTEST_PATH_COLOR, self.line_thickness)


@registry.register_measure
class DistanceToNextGoal(Measure):
    cls_uuid: str = "distance_to_next_goal"
    _sim: Simulator
    _last_pos: Optional[np.ndarray]

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        self._sim = sim
        self._last_pos = None
        self._metric = np.inf
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _compute_distance(self, pos, episode: SequentialEpisode) -> float:
        step_goals = episode.steps[episode._current_step_index].goals
        return self._sim.geodesic_distance(pos, [goal.position for goal in step_goals])

    def reset_metric(self, *args: Any, episode: SequentialEpisode, **kwargs: Any) -> None:
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = self._compute_distance(self._last_pos, episode)

    def update_metric(self, *args: Any, episode: SequentialEpisode, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if np.allclose(self._last_pos, pos):
            return
        self._metric = self._compute_distance(pos, episode)
        self._last_pos = pos


@registry.register_measure
class SequentialSuccess(Measure):
    cls_uuid: str = "sequential_success"
    _radius: float

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self._metric = False
        self._radius = config.SUCCESS_DISTANCE
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: SequentialNavigationTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToNextGoal.cls_uuid])
        self._metric = False

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      task: SequentialNavigationTask, **kwargs: Any) -> None:
        if not task._is_on_last_step(episode):
            return
        d = task.measurements.measures[DistanceToNextGoal.cls_uuid].get_metric()
        if task.is_stop_called and d <= self._radius:
            self._metric = True


@registry.register_measure
class SequentialSPL(Measure):
    _sim: Simulator
    _shortest_dist: float
    _cumul_dist: float
    _last_pos: Optional[np.ndarray]

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        self._sim = sim
        self._shortest_dist = np.inf
        self._cumul_dist = 0.0
        self._last_pos = None
        self._metric = 0.0
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "sequential_spl"

    def _compute_shortest_dist(self, episode: SequentialEpisode) -> float:
        last = [(episode.start_position, 0.0)]
        for step in episode.steps:
            last = [(goal.position, min(d + self._sim.geodesic_distance(pos, goal.position)
                                        for pos, d in last)) for goal in step.goals]
        return min(d for _, d in last)

    def reset_metric(self, *args: Any, episode: SequentialEpisode,
                     task: SequentialNavigationTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [SequentialSuccess.cls_uuid])
        self._shortest_dist = self._compute_shortest_dist(episode)
        self._cumul_dist = 0.0
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = 0.0

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      task: SequentialNavigationTask, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if np.allclose(self._last_pos, pos):
            return
        self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        if task.measurements.measures[SequentialSuccess.cls_uuid].get_metric():
            self._metric = self._shortest_dist / self._cumul_dist
        self._last_pos = pos


@registry.register_measure
class Progress(Measure):
    cls_uuid: str = "progress"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: SequentialNavigationTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [SequentialSuccess.cls_uuid])
        self._metric = 0

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      task: SequentialNavigationTask, **kwargs: Any) -> None:
        success = task.measurements.measures[SequentialSuccess.cls_uuid].get_metric()
        self._metric = 1.0 if success else episode._current_step_index / len(episode.steps)


@registry.register_measure
class PPL(Measure):
    _sim: Simulator
    _shortest_dist: List[float]
    _cumul_dist: float
    _last_pos: Optional[np.ndarray]

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        self._sim = sim
        self._shortest_dist = []
        self._cumul_dist = 0.0
        self._last_pos = None
        self._metric = 0.0
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "ppl"

    def _compute_shortest_dist(self, episode: SequentialEpisode) -> List[float]:
        last = [(episode.start_position, 0.0, None)]
        values = []
        for step in episode.steps:
            values.append(last)
            last = [(g.position, *min((d + self._sim.geodesic_distance(pos, g.position), i)
                                      for i, (pos, d, _) in enumerate(last)))
                    for g in step.goals]
        _, d, back = min(last, key=lambda tup: tup[1])
        distances = []
        while back is not None:
            distances.append(d)
            _, d, back = values.pop()[back]
        distances.reverse()
        return distances

    def reset_metric(self, *args: Any, episode: SequentialEpisode,
                     task: SequentialNavigationTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [Progress.cls_uuid])
        self._shortest_dist = self._compute_shortest_dist(episode)
        self._cumul_dist = 0.0
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = 0.0

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      task: SequentialNavigationTask, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if np.allclose(self._last_pos, pos):
            return
        self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        k = episode._current_step_index
        if k > 0:
            p = task.measurements.measures[Progress.cls_uuid].get_metric()
            d = self._shortest_dist[k - 1]
            self._metric = p * d / max(self._cumul_dist, d)
        self._last_pos = pos
