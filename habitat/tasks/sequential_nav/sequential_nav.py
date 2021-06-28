from typing import Any, Optional, List, Dict

import attr
import numpy as np
from gym import Space, spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, Action, SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Simulator, Sensor, SensorTypes, Observations
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, NavigationEpisode, \
                                  PointGoalSensor, IntegratedPointGoalGPSAndCompassSensor, \
                                  TopDownMap
from habitat.tasks.sequential_nav.utils import make_sequential
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
    success_dist: float

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self.success_dist = config.SUCCESS_DISTANCE
        super().__init__(*args, config=config, **kwargs)

    def reset(self, episode: SequentialEpisode) -> Observations:
        episode._current_step_index = 0
        return super().reset(episode)

    def _check_episode_is_active(self, *args: Any, episode: SequentialEpisode,
                                 **kwargs: Any) -> bool:
        return 0 <= episode._current_step_index < episode.num_steps


@registry.register_task_action
class FoundAction(SimulatorTaskAction):
    name: str = "FOUND"
    _current_episode: Optional[SequentialEpisode] = None

    def reset(self, episode: SequentialEpisode, task: SequentialNavigationTask) -> None:
        self._current_episode = episode

    def step(self, *args: Any, task: SequentialNavigationTask, **kwargs: Any) -> Observations:
        step = self._current_episode.steps[self._current_episode._current_step_index]
        pos = self._sim.get_agent_state().position
        d = self._sim.geodesic_distance(pos, [goal.position for goal in step.goals])
        if d <= task.success_dist or any(goal.radius is not None and d <= goal.radius
                                         for goal in step.goals):
            self._current_episode._current_step_index += 1
        else:
            self._current_episode._current_step_index = -1
        return self._sim.get_observations_at()


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
    _last_step_idx: int
    _last_pos: Optional[np.ndarray]

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        self._sim = sim
        self._last_step_index = 0
        self._last_pos = None
        self._metric = np.inf
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _compute_distance(self, pos, episode: SequentialEpisode) -> float:
        idx = min(episode._current_step_index, episode.num_steps - 1)
        step_goals = episode.steps[idx].goals
        return self._sim.geodesic_distance(pos, [goal.position for goal in step_goals])

    def reset_metric(self, *args: Any, episode: SequentialEpisode, **kwargs: Any) -> None:
        self._last_step_index = episode._current_step_index
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = self._compute_distance(self._last_pos, episode)

    def update_metric(self, *args: Any, episode: SequentialEpisode, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if self._last_step_index == episode._current_step_index \
                and np.allclose(self._last_pos, pos):
                    return
        self._metric = self._compute_distance(pos, episode)
        self._last_step_index = episode._current_step_index
        self._last_pos = pos


@registry.register_measure
class SequentialSuccess(Measure):
    cls_uuid: str = "seq_success"
    _radius: float

    def __init__(self, *args: Any, config: Config, **kwargs: Any) -> None:
        self._metric = 0.0
        self._radius = config.SUCCESS_DISTANCE
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: SequentialNavigationTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToNextGoal.cls_uuid])
        self._metric = 0.0

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      task: SequentialNavigationTask, **kwargs: Any) -> None:
        if episode._current_step_index < episode.num_steps:
            return
        d = task.measurements.measures[DistanceToNextGoal.cls_uuid].get_metric()
        if d <= self._radius:
            self._metric = 1.0


@registry.register_measure
class SequentialSPL(Measure):
    _sim: Simulator
    _shortest_dist: float
    _cumul_dist: float
    _last_step_idx: int
    _last_pos: Optional[np.ndarray]

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        self._sim = sim
        self._shortest_dist = np.inf
        self._cumul_dist = 0.0
        self._last_step_idx = 0
        self._last_pos = None
        self._metric = 0.0
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "seq_spl"

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
        self._last_step_idx = 0
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = 0.0

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      task: SequentialNavigationTask, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if self._last_step_idx == episode._current_step_index \
                and np.allclose(self._last_pos, pos):
                    return
        self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        if task.measurements.measures[SequentialSuccess.cls_uuid].get_metric():
            self._metric = self._shortest_dist / max(self._shortest_dist, self._cumul_dist)
        self._last_step_idx = episode._current_step_index
        self._last_pos = pos


@registry.register_measure
class Progress(Measure):
    cls_uuid: str = "progress"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0.0
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: SequentialNavigationTask, **kwargs: Any) -> None:
        self._metric = 0.0

    def update_metric(self, *args: Any, episode: SequentialEpisode, **kwargs: Any) -> None:
        if episode._current_step_index > 0:
            self._metric = episode._current_step_index / episode.num_steps


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
