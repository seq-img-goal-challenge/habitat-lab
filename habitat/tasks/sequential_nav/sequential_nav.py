from typing import Any, Optional, List

import attr
import numpy as np
from gym import Space, spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import EmbodiedTask, Measure, Action
from habitat.core.registry import registry
from habitat.core.simulator import Simulator, Sensor, SensorTypes, Observations
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask


@attr.s(auto_attribs=True, kw_only=True)
class SequentialStep:
    goals: List[NavigationGoal] = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class SequentialEpisode(Episode):
    steps: List[SequentialStep] = attr.ib(default=None, validator=not_none_validator)
    _current_step_index: int = attr.ib(init=False, default=0)

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._current_step_index = 0


@attr.s(auto_attribs=True, kw_only=True)
class SequentialDataset(Dataset):
    episodes: List[SequentialEpisode] = attr.ib(default=None, validator=not_none_validator)

    def get_max_sequence_len(self):
        return max(len(episode.steps) for episode in self.episodes)


@registry.register_task(name="SequentialNav-v0")
class SequentialNavigationTask(NavigationTask):
    is_stop_called: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.is_stop_called = False
        super().__init__(*args, **kwargs)

    def reset(self, episode: SequentialEpisode) -> Observations:
        self.is_stop_called = False
        episode._current_step_index = 0
        return super().reset(episode)

    def _is_on_last_step(self, episode: SequentialEpisode):
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
                if any(d <= goal.radius for goal in step.goals):
                    episode._current_step_index += 1
                    self.is_stop_called = False
                    return True
                else:
                    return False
        else:
            return True


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
class SequentialProgress(Measure):
    cls_uuid: str = "sequential_progress"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: SequentialNavigationTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [SequentialSuccess.cls_uuid])
        self._metric = 0

    def update_metric(self, *args: Any, episode: SequentialEpisode,
                      **kwargs: Any) -> None:
        success = task.measurements.measures[SequentialSuccess.cls_uuid].get_metric()
        self._metric = 1.0 if success else episode._current_step_index / len(episode.steps)


@registry.register_measure
class SequentialSPL(Measure):
    cls_uuid: str = "sequential_spl"
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
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _compute_shortest_dist(self, episode: SequentialEpisode):
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
