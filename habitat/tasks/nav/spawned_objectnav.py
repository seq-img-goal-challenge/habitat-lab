from typing import List, Dict, Any, Optional
import attr

import numpy as np
import quaternion
import magnum as mn
from gym import Space, spaces

from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator, Sensor, SensorTypes, \
                                   VisualObservation, Observations
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.config import Config
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, DistanceToGoal
from habitat.datasets.spawned_objectnav.utils import get_uniform_view_pt_positions, \
                                                     get_view_pt_rotations, \
                                                     render_view_pts
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType, ManagedRigidObject
from habitat_sim.attributes_managers import ObjectAttributesManager


@attr.s(auto_attribs=True, kw_only=True)
class SpawnedObjectGoal(NavigationGoal):
    # Inherited
    # position: List[float]
    # radius: Optional[float]
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    object_template_id: str
    valid_view_points_indices: np.ndarray = attr.ib(default=None, validator=not_none_validator)
    valid_view_points_ious: Optional[np.ndarray] = None
    _spawned_object: Optional[ManagedRigidObject] = attr.ib(init=False, default=None)
    _rotated_bb: Optional[np.ndarray] = attr.ib(init=False, default=None)

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._spawned_object_id = None
        self._rotated_bb = None

    def _spawn_in_sim(self, sim: Simulator) -> None:
        obj_mngr = sim.get_rigid_object_manager()
        self._spawned_object = obj_mngr.add_object_by_template_handle(self.object_template_id)
        self._set_bb(self._spawned_object.root_scene_node.cumulative_bb)
        shift = -self._rotated_bb[:, 1].min()
        self._spawned_object.translation = mn.Vector3(*self.position) + mn.Vector3(0, shift, 0)
        self._spawned_object.rotation = mn.Quaternion(self.rotation[:3], self.rotation[3])
        self._spawned_object.motion_type = MotionType.STATIC

    def _despawn_from_sim(self, sim: Simulator) -> None:
        self._rotated_bb = None

        obj_mngr = sim.get_rigid_object_manager()
        obj_mngr.remove_object_by_id(self._spawned_object.object_id)
        del self._spawned_object
        self._spawned_object = None

    def _set_bb(self, bb: mn.Range3D) -> None:
        rot = np.quaternion(self.rotation[3], *self.rotation[:3])
        bb_pts = np.array([bb.back_bottom_left, bb.back_bottom_right,
                           bb.front_bottom_right, bb.front_bottom_left,
                           bb.back_top_left, bb.back_top_right,
                           bb.front_top_right, bb.front_top_left])
        bb_q = quaternion.from_float_array(np.concatenate((np.zeros((8, 1)), bb_pts), -1))
        self._rotated_bb = quaternion.as_float_array(rot * bb_q * rot.conj())[:, 1:]

    @property
    def pathfinder_targets(self) -> List[List[float]]:
        return (self.position + self._rotated_bb).tolist()


@attr.s(auto_attribs=True, kw_only=True)
class SpawnedObjectNavEpisode(Episode):
    # Inherited
    # episode_id: str
    # scene_id: str
    # start_position: List[float]
    # start_rotation: List[float]
    # info: Dict[str, Any]
    # _shortest_path_cache: Any
    object_category: str
    object_category_index: int
    goals: List[SpawnedObjectGoal] = attr.ib(default=None, validator=not_none_validator)

    @property
    def all_goals(self) -> List[SpawnedObjectGoal]:
        return self.goals


@registry.register_task(name="SpawnedObjectNav-v0")
class SpawnedObjectNavTask(NavigationTask):
    _current_episode: Optional[SpawnedObjectNavEpisode] = None
    is_stop_called: bool = False

    def _check_templates(self) -> bool:
        tmpl_mngr = self._sim.get_object_template_manager()
        loaded = set(tmpl_mngr.get_file_template_handles())
        return all(goal.object_template_id in loaded
                   for goal in self._current_episode.all_goals)

    def _preload_templates(self) -> None:
        tmpl_mngr = self._sim.get_object_template_manager()
        to_load = self._dataset.get_episode_iterator().get_object_subset()
        for tmpl_path in to_load:
            tmpl_mngr.load_configs(tmpl_path)

    def _despawn_objects(self) -> None:
        for goal in self._current_episode.all_goals:
            goal._despawn_from_sim(self._sim)

    def _spawn_objects(self) -> None:
        for goal in self._current_episode.all_goals:
            goal._spawn_in_sim(self._sim)

    def _recompute_navmesh_for_static_objects(self):
        settings = NavMeshSettings()
        settings.set_defaults()
        ag_cfg = getattr(self._sim.habitat_config,
                         f"AGENT_{self._sim.habitat_config.DEFAULT_AGENT_ID}",
                         self._sim.habitat_config.AGENT_0)
        settings.agent_radius = ag_cfg.RADIUS
        settings.agent_height = ag_cfg.HEIGHT
        self._sim.recompute_navmesh(self._sim.pathfinder, settings, True)

    def reset(self, episode: SpawnedObjectNavEpisode) -> Observations:
        if self._current_episode is not None \
                and self._current_episode.scene_id == episode.scene_id:
            self._despawn_objects()
        self._current_episode = episode
        if not self._check_templates():
            self._preload_templates()
        self._spawn_objects()
        if self._config.ENABLE_OBJECT_COLLISIONS:
            self._recompute_navmesh_for_static_objects()
        return super().reset(episode)


@registry.register_sensor
class SpawnedObjectGoalCategorySensor(Sensor):
    _max_object_category_index: int

    def __init__(self, config: Config, dataset: "SpawnedObjectNavDatasetV1",
                 *args: Any, **kwargs: Any) -> None:
        self._max_object_category_index = dataset.get_max_object_category_index()
        super().__init__(config=config, *args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "objectgoal_category"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=self._max_object_category_index,
                          shape=(1,), dtype=np.int64)

    def get_observation(self, episode: SpawnedObjectNavEpisode, *args, **kwargs) -> np.ndarray:
        return np.array([episode.object_category_index], dtype=np.int64)


@registry.register_sensor
class SpawnedObjectGoalAppearanceSensor(Sensor):
    _sim: Simulator
    _sim_sensor: Sensor
    _oob_pos: np.ndarray
    _view_pts_rel_positions: np.ndarray
    _view_pts_rotations: np.ndarray
    _cached_ep_id: Optional[str]
    _cached_appearance: Optional[np.ndarray]

    def __init__(self, config: Config, sim: Simulator, task: SpawnedObjectNavTask,
                       *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        if config.SENSOR_UUID is None:
            try:
                sensor_type = SensorTypes[config.SENSOR_TYPE]
            except KeyError as e:
                raise RuntimeError(f"Invalid sensor type {config.SENSOR_TYPE}.") from e
            try:
                self._sim_sensor = next((sensor for sensor in sim.sensor_suite.sensors.values()
                                         if sensor.sensor_type == sensor_type))
            except StopIteration as e:
                raise RuntimeError(f"Could not find a sensor of type {config.SENSOR_TYPE} "
                                   + "in simulator sensor suite.") from e
        else:
            try:
                self._sim_sensor = sim.sensor_suite.sensors[config.SENSOR_UUID]
            except KeyError as e:
                raise RuntimeError(f"Could not find sensor {config.SENSOR_UUID} "
                                   + "in simulator sensor suite.") from e
        _, (_, max_y, _) = sim.pathfinder.get_bounds()
        self._oob_pos = np.array([0.0, 2 * max_y, 0.0])
        self._view_pts_rel_positions = get_uniform_view_pt_positions(
            config.VIEW_POINTS.NUM_ANGLES, config.VIEW_POINTS.NUM_RADII,
            config.VIEW_POINTS.MIN_RADIUS, config.VIEW_POINTS.MAX_RADIUS
        )
        self._view_pts_rel_positions += np.array(self._sim_sensor.config.POSITION)
        self._view_pts_rotations = get_view_pt_rotations(self._view_pts_rel_positions)
        self._cached_ep_id = None
        self._cached_appearance = None
        super().__init__(config=config, *args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "objectgoal_appearance"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return self._sim_sensor.sensor_type

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        src_space = self._sim_sensor.observation_space
        extended_shape = (self.config.NUM_VIEWS,) + src_space.shape
        return spaces.Box(low=np.broadcast_to(src_space.low, extended_shape),
                          high=np.broadcast_to(src_space.high, extended_shape),
                          dtype=src_space.dtype)

    def _move_object_out_of_context(self, goal: SpawnedObjectGoal):
        goal._spawned_object.motion_type = MotionType.KINEMATIC
        goal._spawned_object.translation = self._oob_pos

        if self.config.RANDOM_OBJECT_ROTATION == "3D":
            ax = np.random.randn(3)
            ax /= np.linalg.norm(ax)
            a = 2 * np.pi * np.random.random()
            rot = mn.Quaternion.rotation(mn.Rad(a), mn.Vector3(*ax))
            goal._spawned_object.rotation = rot

    def _restore_object_in_context(self, goal: SpawnedObjectGoal):
        goal._spawned_object.translation = goal.position
        if self.config.RANDOM_OBJECT_ROTATION == "3D":
            goal._spawned_object.rotation = mn.Quaternion(goal.rotation[:3], goal.rotation[3])
        goal._spawned_object.motion_type = MotionType.STATIC

    def _render_views_around_goal(self, goal: SpawnedObjectGoal, num_views: int) -> None:
        if self.config.OUT_OF_CONTEXT:
            self._move_object_out_of_context(goal)
            indices = np.random.permutation(self._view_pts_rel_positions.shape[0])[:num_views]
            positions = self._oob_pos + self._view_pts_rel_positions[indices]
        else:
            indices = np.random.choice(goal.valid_view_points_indices, num_views, True)
            positions = np.array(goal.position) + self._view_pts_rel_positions[indices]
        rotations = self._view_pts_rotations[indices]
        views = render_view_pts(self._sim, positions, rotations)
        if self.config.OUT_OF_CONTEXT:
            self._restore_object_in_context(goal)
        return views[self._sim_sensor.uuid]

    def get_observation(self, episode: SpawnedObjectNavEpisode,
                        *args, **kwargs) -> VisualObservation:
        if self._cached_ep_id is None or self._cached_ep_id != episode.episode_id:
            views = []
            num_views_per_goal, more_views = divmod(self.config.NUM_VIEWS, len(episode.goals))
            for k, goal in enumerate(episode.goals):
                num_views = num_views_per_goal + (1 if k < more_views else 0)
                views.append(self._render_views_around_goal(goal, num_views))
            self._cached_ep_id = episode.episode_id
            self._cached_appearance = np.concatenate(views, 0)
        return self._cached_appearance


@registry.register_measure
class DistanceToObject(Measure):
    _sim: Simulator
    _metric: float
    _last_pos: Optional[np.ndarray]
    _ep_targets: List[List[float]]

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._sim = sim
        self._last_pos = None
        self._ep_targets = []

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return DistanceToGoal.cls_uuid

    def reset_metric(self, episode: SpawnedObjectNavEpisode,
                     *args: Any, **kwargs: Any) -> None:
        self._ep_targets = sum((goal.pathfinder_targets for goal in episode.goals), [])
        self._last_pos = self._sim.get_agent_state().position
        self._metric = self._sim.geodesic_distance(self._last_pos, self._ep_targets)

    def update_metric(self, episode: SpawnedObjectNavEpisode,
                      *args: Any, **kwargs: Any) -> None:
        pos = self._sim.get_agent_state().position
        if np.allclose(pos, self._last_pos):
            return
        self._metric = self._sim.geodesic_distance(pos, self._ep_targets)
        self._last_pos = pos
