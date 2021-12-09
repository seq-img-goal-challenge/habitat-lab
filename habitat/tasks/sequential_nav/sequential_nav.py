from typing import Any, Optional, List, Dict, ClassVar

import attr
import numpy as np
import cv2
import gym

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
from habitat.utils.visualizations import maps, fog_of_war


@attr.s(auto_attribs=True, kw_only=True)
class SequentialStep:
    goals: List[NavigationGoal] = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class SequentialEpisode(Episode):
    steps: List[SequentialStep] = attr.ib(default=None, validator=not_none_validator)
    _current_step_index: int = attr.ib(init=False, default=0)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def goals(self) -> List[NavigationGoal]:
        return self.steps[self._current_step_index].goals

    @property
    def all_goals(self) -> List[NavigationGoal]:
        return sum((step.goals for step in self.steps), [])

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
    def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, config=config, **kwargs)

    def reset(self, episode: SequentialEpisode) -> Observations:
        episode._current_step_index = 0
        return super().reset(episode)

    def _check_episode_is_active(self, episode: SequentialEpisode,
                                       *args: Any, **kwargs: Any) -> bool:
        return 0 <= episode._current_step_index < episode.num_steps


@registry.register_task_action
class FoundAction(SimulatorTaskAction):
    name: ClassVar[str] = "FOUND"
    _current_episode: Optional[SequentialEpisode] = None

    def reset(self, episode: SequentialEpisode, task: SequentialNavigationTask) -> None:
        self._current_episode = episode

    def step(self, task: SequentialNavigationTask, *args: Any, **kwargs: Any) -> Observations:
        step = self._current_episode.steps[self._current_episode._current_step_index]
        step_targets = sum((goal.pathfinder_targets for goal in step.goals), [])
        pos = self._sim.get_agent_state().position
        d = self._sim.geodesic_distance(pos, step_targets)
        if d <= task._config.SUCCESS_DISTANCE:
            self._current_episode._current_step_index += 1
        else:
            self._current_episode._current_step_index = -1
        return self._sim.get_observations_at()


SequentialPointGoalSensor = make_sequential(PointGoalSensor)
SequentialOnlinePointGoalSensor = make_sequential(IntegratedPointGoalGPSAndCompassSensor,
                                                  name="SequentialOnlinePointGoalSensor")

@registry.register_sensor
class SequentialMapSensor(Sensor):
    _sim: Simulator
    _last_ep_id: Optional[str]
    _last_step_idx: Optional[int]
    _topdown_map: Optional[np.ndarray]
    _fog: Optional[np.ndarray]
    _origin: Optional[np.ndarray]
    _mppx: Optional[float]

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._last_ep_id = None
        self._last_step_idx = None
        self._topdown_map = None
        self._fog = None
        self._origin = None
        self._mppx = None
        super().__init__(*args, sim=sim, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Space:
        res = self.config.RESOLUTION
        return gym.spaces.Box(0, 255, (res, res), np.uint8)

    def get_observation(self, episode: SequentialEpisode,
                              *args: Any, **kwargs: Any) -> np.ndarray:
        mrk = self.config.MARKER_SIZE // 2
        if self._last_ep_id != episode.episode_id:
            res = self.config.RESOLUTION
            self._origin, upper = self._sim.pathfinder.get_bounds()
            mppx = (upper - self._origin) / res
            self._mppx = max(mppx[0], mppx[2])
            altitude = self._sim.get_agent_state().position[1]
            nav_mask = self._sim.pathfinder.get_topdown_view(self._mppx, altitude)
            h, w = nav_mask.shape
            self._topdown_map = np.full((res, res), maps.MAP_INVALID_POINT, dtype=np.uint8)
            oi, oj = (res - h) // 2, (res - w) // 2
            self._topdown_map[oi:oi + h, oj:oj + w][nav_mask] = maps.MAP_VALID_POINT
            self._origin[0] -= oj * self._mppx
            self._origin[1] = altitude
            self._origin[2] -= oi * self._mppx
            self._fog = np.zeros_like(self._topdown_map)

        if self._last_ep_id != episode.episode_id \
                or self._last_step_idx != episode._current_step_index:
            for t, step in enumerate(episode.steps):
                for goal in step.goals:
                    j, _, i = ((goal.position - self._origin) / self._mppx).astype(np.int64)
                    if t == episode._current_step_index:
                        indic = maps.MAP_NEXT_TARGET_POINT_INDICATOR
                    else:
                        indic = maps.MAP_TARGET_POINT_INDICATOR
                    cv2.circle(self._topdown_map, (j, i), mrk, indic, -1)
            self._last_ep_id = episode.episode_id
            self._last_step_idx = episode._current_step_index

        s = self._sim.get_agent_state()
        j, _, i = ((s.position - self._origin) / self._mppx).astype(np.int64)
        a = np.pi + 2 * np.arctan(s.rotation.y / s.rotation.w)
        topdown_map = self._topdown_map.copy()
        if self.config.FOG_OF_WAR:
            fov = self.config.HFOV
            d = self.config.VISIBILITY
            self._fog = fog_of_war.reveal_fog_of_war(self._topdown_map, self._fog,
                                                     np.array((i, j)), a, fov, d / self._mppx)
            topdown_map[self._fog == 0] = maps.MAP_INVALID_POINT
        cv2.circle(topdown_map, (j, i), mrk, maps.MAP_SOURCE_POINT_INDICATOR, -1)
        return topdown_map


@registry.register_sensor
class SequentialEgoMapSensor(Sensor):
    _sim: Simulator
    _last_ep_id: Optional[str]
    _last_step_idx: Optional[int]
    _topdown_map: Optional[np.ndarray]
    _fog: Optional[np.ndarray]

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._last_ep_id = None
        self._last_step_idx = None
        self._topdown_map = None
        self._fog = None
        super().__init__(*args, sim=sim, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "ego_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> gym.spaces.Space:
        res = int(2 * self.config.VISIBILITY / self.config.METERS_PER_PIXEL)
        return gym.spaces.Box(0, 255, (res, res), np.uint8)

    def get_observation(self, episode: SequentialEpisode,
                              *args: Any, **kwargs: Any) -> np.ndarray:
        mrk = self.config.MARKER_SIZE // 2
        mppx = self.config.METERS_PER_PIXEL
        if self._last_ep_id != episode.episode_id:
            self._topdown_map = maps.get_topdown_map_from_sim(self._sim, meters_per_pixel=mppx)
            self._fog = np.zeros_like(self._topdown_map)

        if self._last_ep_id != episode.episode_id \
                or self._last_step_idx != episode._current_step_index:
            for t, step in enumerate(episode.steps):
                for goal in step.goals:
                    i, j = maps.to_grid(goal.position[2], goal.position[0],
                                        self._topdown_map.shape, self._sim)
                    if t == episode._current_step_index:
                        indic = maps.MAP_NEXT_TARGET_POINT_INDICATOR
                    else:
                        indic = maps.MAP_TARGET_POINT_INDICATOR
                    cv2.circle(self._topdown_map, (j, i), mrk, indic, -1)
            self._last_ep_id = episode.episode_id
            self._last_step_idx = episode._current_step_index

        s = self._sim.get_agent_state()
        i, j = maps.to_grid(s.position[2], s.position[0], self._topdown_map.shape, self._sim)
        a = 2 * np.arctan(s.rotation.y / s.rotation.w)
        d = self.config.VISIBILITY
        topdown_map = self._topdown_map.copy()
        if self.config.FOG_OF_WAR:
            fov = self.config.HFOV
            self._fog = fog_of_war.reveal_fog_of_war(self._topdown_map, self._fog,
                                                     np.array((i, j)), np.pi + a,
                                                     fov, d / mppx)
            topdown_map[self._fog == 0] = maps.MAP_INVALID_POINT

        h, w = topdown_map.shape
        rot = cv2.getRotationMatrix2D((j, i), -np.degrees(a), 1.0)
        rot_h = int(h * abs(rot[0, 0]) + w * abs(rot[0, 1]))
        rot_w = int(h * abs(rot[0, 1]) + w * abs(rot[0, 0]))
        rot[0, 2] += 0.5 * rot_w - j
        rot[1, 2] += 0.5 * rot_h - i
        rot_map = cv2.warpAffine(topdown_map, rot, (rot_w, rot_h),
                                 borderValue=maps.MAP_INVALID_POINT)
        res = int(2 * d / mppx)
        ego_map = np.zeros((res, res), dtype=np.uint8)
        oi, oj = max(0, (res - rot_h) // 2), max(0, (res - rot_w) // 2)
        y, x = max(0, (rot_h - res) // 2), max(0, (rot_w - res) // 2)
        ego_map[oi:oi + rot_h, oj: oj + rot_w] = rot_map[y:y + res, x: x + res]
        return ego_map


@registry.register_measure
class SequentialTopDownMap(TopDownMap):
    _last_step_index: int = 0

    def _compute_shortest_path(self, episode: SequentialEpisode,
                                     start_pos: List[float]) -> List[List[float]]:
        last = [(start_pos, 0.0, None)]
        values = []
        for step in episode.steps[episode._current_step_index:]:
            values.append(last)
            last = [(trg, *min((d + self._sim.geodesic_distance(pos, trg), i)
                               for i, (pos, d, _) in enumerate(last)))
                    for goal in step.goals for trg in goal.pathfinder_targets]

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
        if self._config.DRAW_GOAL_POSITIONS:
            for t, step in enumerate(episode.steps):
                if t == episode._current_step_index:
                    for goal in step.goals:
                        if self._is_on_same_floor(goal.position[1]):
                            super()._draw_point(goal.position,
                                                maps.MAP_NEXT_TARGET_POINT_INDICATOR)
                else:
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

    def reset_metric(self, episode: SequentialEpisode, *args: Any, **kwargs: Any) -> None:
        self._last_step_index = episode._current_step_index
        super().reset_metric(*args, episode=episode, **kwargs)

    def update_metric(self, episode: SequentialEpisode, *args: Any, **kwargs: Any) -> None:
        if episode._current_step_index != self._last_step_index:
            self._draw_goals_positions(episode)
            self._last_step_index = episode._current_step_index
        super().update_metric(*args, episode=episode, **kwargs)


@registry.register_measure
class DistanceToNextGoal(Measure):
    cls_uuid: ClassVar[str] = "distance_to_next_goal"
    _sim: Simulator
    _last_step_index: int
    _last_pos: Optional[np.ndarray]

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._last_step_index = 0
        self._last_pos = None
        self._metric = np.inf
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _compute_distance(self, pos: np.ndarray, episode: SequentialEpisode) -> float:
        idx = np.clip(episode._current_step_index,
                      self._last_step_index, episode.num_steps - 1)
        step_targets = sum((goal.pathfinder_targets for goal in episode.steps[idx].goals), [])
        return self._sim.geodesic_distance(pos, step_targets)

    def reset_metric(self, episode: SequentialEpisode, *args: Any, **kwargs: Any) -> None:
        self._last_step_index = episode._current_step_index
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = self._compute_distance(self._last_pos, episode)

    def update_metric(self, episode: SequentialEpisode, *args: Any, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if self._last_step_index == episode._current_step_index \
                and np.allclose(self._last_pos, pos):
                    return
        self._metric = self._compute_distance(pos, episode)
        self._last_step_index = episode._current_step_index
        self._last_pos = pos


@registry.register_measure
class SequentialSuccess(Measure):
    cls_uuid: ClassVar[str] = "seq_success"
    _config: Config

    def __init__(self, config: Config, *args: Any, **kwargs: Any) -> None:
        self._config = config
        self._metric = 0.0
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: SequentialNavigationTask, *args: Any, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToNextGoal.cls_uuid])
        self._metric = 0.0

    def update_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                            *args: Any, **kwargs: Any) -> None:
        if episode._current_step_index < episode.num_steps:
            return
        d = task.measurements.measures[DistanceToNextGoal.cls_uuid].get_metric()
        if d <= self._config.SUCCESS_DISTANCE:
            self._metric = 1.0


@registry.register_measure
class SequentialSPL(Measure):
    _sim: Simulator
    _shortest_dist: float
    _cumul_dist: float
    _last_step_index: int
    _last_pos: Optional[np.ndarray]

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._shortest_dist = np.inf
        self._cumul_dist = 0.0
        self._last_step_index = 0
        self._last_pos = None
        self._metric = 0.0
        super().__init__(*args, sim=sim, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "seq_spl"

    def _compute_shortest_dist(self, episode: SequentialEpisode) -> float:
        last = [(episode.start_position, 0.0)]
        for step in episode.steps:
            last = [(trg, min(d + self._sim.geodesic_distance(pos, trg)
                              for pos, d in last))
                    for goal in step.goals for trg in goal.pathfinder_targets]
        return min(d for _, d in last)

    def reset_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                           *args: Any, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [SequentialSuccess.cls_uuid])
        self._shortest_dist = self._compute_shortest_dist(episode)
        self._cumul_dist = 0.0
        self._last_step_index = episode._current_step_index
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = 0.0

    def update_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                            *args: Any, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        if self._last_step_index == episode._current_step_index \
                and np.allclose(self._last_pos, pos):
                    return
        self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        if task.measurements.measures[SequentialSuccess.cls_uuid].get_metric():
            self._metric = self._shortest_dist / max(self._shortest_dist, self._cumul_dist)
        self._last_pos = pos
        self._last_step_index = episode._current_step_index


@registry.register_measure
class Progress(Measure):
    cls_uuid: ClassVar[str] = "progress"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0.0
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, task: SequentialNavigationTask, *args: Any, **kwargs: Any) -> None:
        self._metric = 0.0

    def update_metric(self, episode: SequentialEpisode, *args: Any, **kwargs: Any) -> None:
        if episode._current_step_index > 0:
            self._metric = episode._current_step_index / episode.num_steps


@registry.register_measure
class PPL(Measure):
    _sim: Simulator
    _shortest_dist: List[float]
    _cumul_dist: float
    _last_step_index: int
    _last_pos: Optional[np.ndarray]

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._shortest_dist = []
        self._cumul_dist = 0.0
        self._last_step_index = 0
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
            last = [(trg, *min((d + self._sim.geodesic_distance(pos, trg), i)
                               for i, (pos, d, _) in enumerate(last)))
                    for goal in step.goals for trg in goal.pathfinder_targets]
        _, d, back = min(last, key=lambda tup: tup[1])
        distances = []
        while back is not None:
            distances.append(d)
            _, d, back = values.pop()[back]
        distances.reverse()
        return distances

    def reset_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                           *args: Any, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [Progress.cls_uuid])
        self._shortest_dist = self._compute_shortest_dist(episode)
        self._cumul_dist = 0.0
        self._last_step_index = 0
        self._last_pos = np.array(self._sim.get_agent_state().position)
        self._metric = 0.0

    def update_metric(self, episode: SequentialEpisode, task: SequentialNavigationTask,
                            *args: Any, **kwargs: Any) -> None:
        pos = np.array(self._sim.get_agent_state().position)
        idx = episode._current_step_index
        if self._last_step_index == idx and np.allclose(self._last_pos, pos):
            return
        self._cumul_dist += np.linalg.norm(pos - self._last_pos)
        if idx > 0:
            p = task.measurements.measures[Progress.cls_uuid].get_metric()
            d = self._shortest_dist[idx - 1]
            self._metric = p * d / max(self._cumul_dist, d)
        self._last_step_index = idx
        self._last_pos = pos
