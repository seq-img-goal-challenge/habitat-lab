from typing import List, Dict, Any, Optional
import attr
from copy import deepcopy
from itertools import chain

import numpy as np
import quaternion
import magnum as mn
from gym import Space, spaces

from habitat.core.dataset import Episode
from habitat.core.simulator import Simulator, Sensor, SensorTypes, \
                                   VisualObservation, Observations
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.config import Config
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat.datasets.spawned_objectnav.utils import get_random_view_pt_positions, \
                                                     get_view_pt_rotations, \
                                                     render_view_pts
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


@attr.s(auto_attribs=True, kw_only=True)
class ViewPoint:
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    iou: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class SpawnedObjectGoal(NavigationGoal):
    # Inherited
    # position: List[float]
    # radius: Optional[float]
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    view_points: List[ViewPoint] = attr.ib(default=None, validator=not_none_validator)
    object_template_id: str
    _spawned_object_id: Optional[int] = attr.ib(init=False, default=None)

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._spawned_object_id = None


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


@registry.register_task(name="SpawnedObjectNav-v0")
class SpawnedObjectNavTask(NavigationTask):
    _template_manager: "ObjectAttributesManager" # from habitat_sim bindings...
    _loaded_object_templates: Dict[str, int] # to avoid relying on tmpl_mngr object handles...
    is_stop_called: bool

    def __init__(self, config: Config, sim: Simulator,
                 dataset: Optional["SpawnedObjectNavDatasetV1"]=None) -> None:
        self._template_manager = sim.get_object_template_manager()
        self._loaded_object_templates = {}
        self.is_stop_called = False
        super().__init__(config=config, sim=sim, dataset=dataset)

    def _reload_templates(self, episode: SpawnedObjectNavEpisode) -> None:
        loaded = set(self._loaded_object_templates)
        to_load = self._dataset.get_objects_to_load(episode)
        for tmpl_id in loaded - to_load:
            mngr_id = self._loaded_object_templates[tmpl_id]
            self._template_manager.remove_template_by_ID(mngr_id)
            del self._loaded_object_templates[tmpl_id]
        for tmpl_id in to_load - loaded:
            mngr_id, = self._template_manager.load_configs(tmpl_id)
            self._loaded_object_templates[tmpl_id] = mngr_id

    def _despawn_objects(self) -> None:
        for obj_id in self._sim.get_existing_object_ids():
            self._sim.remove_object(obj_id)

    def _spawn_objects(self, episode: SpawnedObjectNavEpisode) -> None:
        for goal in episode.goals:
            mngr_id = self._loaded_object_templates[goal.object_template_id]
            goal._spawned_object_id = self._sim.add_object(mngr_id)
            self._sim.set_translation(goal.position, goal._spawned_object_id)
            self._sim.set_rotation(mn.Quaternion(goal.rotation[:3], goal.rotation[3]),
                                   goal._spawned_object_id)
            self._sim.set_object_motion_type(MotionType.STATIC, goal._spawned_object_id)

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
        self._despawn_objects()
        self._reload_templates(episode)
        self._spawn_objects(episode)
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
    _sensor_uuid: "str"
    _oob_pos: np.ndarray
    _cache_ep_id: Optional[str]
    _appearance_cache: Optional[np.ndarray]

    #TODO (gbono): Make appearance RGB-D? (currently RGB only, like imagegoal)
    def __init__(self, config: Config, sim: Simulator, *args, **kwargs) -> None:
        self._sim = sim
        self._sensor_uuid = next((uuid for uuid, sensor in sim.sensor_suite.sensors.items()
                                  if sensor.sensor_type == SensorTypes.COLOR), None)
        if self._sensor_uuid is None:
            raise RuntimeError("Could not find a sensor of type 'COLOR'" \
                               + "in the simulator sensor suite")
        max_y = sim.pathfinder.get_bounds()[1][1]
        self._oob_pos = np.array([0.0, max_y + 50.0, 0.0])
        self._cache_ep_id = None
        self._appearance_cache = None
        super().__init__(config=config, *args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "objectgoal_appearance"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        src_space = self._sim.sensor_suite.observation_spaces.spaces[self._sensor_uuid]
        extended_shape = (self.config.NUM_VIEWS,) + src_space.shape
        return spaces.Box(low=np.broadcast_to(src_space.low, extended_shape),
                          high=np.broadcast_to(src_space.high, extended_shape),
                          dtype=src_space.dtype)

    def _move_object_out_of_context(self, goal: SpawnedObjectGoal):
        self._sim.set_object_motion_type(MotionType.KINEMATIC, goal._spawned_object_id)
        self._sim.set_translation(self._oob_pos, goal._spawned_object_id)

        if self.config.RANDOM_OBJECT_ROTATION == "3D":
            ax = np.random.randn(3)
            ax /= np.linalg.norm(ax)
            a = 2 * np.pi * np.random.random()
            rot = mn.Quaternion.rotation(mn.Rad(a), mn.Vector3(*ax))
            self._sim.set_rotation(rot, goal._spawned_object_id)

    def _restore_object_in_context(self, goal: SpawnedObjectGoal):
        self._sim.set_translation(goal.position, goal._spawned_object_id)
        if self.config.RANDOM_OBJECT_ROTATION == "3D":
            self._sim.set_rotation(mn.Quaternion(goal.rotation[:3], goal.rotation[3]),
                                   goal._spawned_object_id)
        self._sim.set_object_motion_type(MotionType.STATIC, goal._spawned_object_id)

    def _render_views_around_goal(self, goal: SpawnedObjectGoal, num_views: int) -> None:
        if self.config.OUT_OF_CONTEXT:
            self._move_object_out_of_context(goal)

            rel_positions = get_random_view_pt_positions(num_views)
            rotations = get_view_pt_rotations(rel_positions)
            views = render_view_pts(self._sim, self._oob_pos + rel_positions, rotations)

            self._restore_object_in_context(goal)
        else:
            view_points = np.random.choice(goal.view_points, num_views, True)
            positions = np.array([view_pt.position for view_pt in view_points])
            rotations = np.array([view_pt.rotation for view_pt in view_points])
            rotations = quaternion.from_float_array(rotations[:, [3, 0, 1, 2]])
            views = render_view_pts(self._sim, positions, rotations)
        return views['rgb'][..., :3]

    def get_observation(self, episode: SpawnedObjectNavEpisode,
                        *args, **kwargs) -> VisualObservation:
        if self._cache_ep_id is None or self._cache_ep_id != episode.episode_id:
            views = []
            num_views_per_goal, more_views = divmod(self.config.NUM_VIEWS, len(episode.goals))
            for k, goal in enumerate(episode.goals):
                num_views = num_views_per_goal + (1 if k < more_views else 0)
                views.append(self._render_views_around_goal(goal, num_views))
            self._cache_ep_id = episode.episode_id
            self._appearance_cache = np.concatenate(views, 0)
        return self._appearance_cache
