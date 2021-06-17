from typing import List, Dict, Any, Optional
import attr
from copy import deepcopy
from itertools import chain

import numpy as np
import magnum as mn
from gym import Space, spaces

from habitat.core.dataset import Episode
from habitat.core.simulator import Simulator, Sensor, SensorTypes, \
                                   VisualObservation, Observations
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.config import Config
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask


@attr.s(auto_attribs=True, kw_only=True)
class SpawnedObjectGoal(NavigationGoal):
    # TODO(gbono): Inherit from ObjectGoal?
    # Inherited
    # position: List[float]
    # radius: float
    orientation: Optional[List[float]] = attr.ib(default=None, validator=not_none_validator)
    object_template_id: str
    _spawned_object_id: Optional[int] = attr.ib(init=False, default=None)
    _appearance_cache: Optional[List[VisualObservation]] = attr.ib(init=False, default=None)

    def __getstate__(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._spawned_object_id = None
        self._appearance_cache = None


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
    #TODO (gbono): Make appearance RGB-D? (currently RGB only, like imagegoal)
    def __init__(self, config: Config, sim: Simulator, *args, **kwargs) -> None:
        self._sim = sim
        self._sensor_uuid = next((uuid for uuid, sensor in sim.sensor_suite.sensors.items()
                                  if sensor.sensor_type == SensorTypes.COLOR), None)
        if self._sensor_uuid is None:
            raise RuntimeError("Could not find a sensor of type 'COLOR'" \
                               + "in the simulator sensor suite")
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

    def _generate_views_around_goal(self, goal: SpawnedObjectGoal, num_views:int) -> None:
        goal_pos = self.config.OUT_OF_CONTEXT_POS if self.config.OUT_OF_CONTEXT \
                else goal.position
        goal._appearance_cache = []
        while len(goal._appearance_cache) < num_views: #TODO(gbono): limit number of cand?
            if self.config.OUT_OF_CONTEXT and self.config.RANDOM_OBJECT_ORIENTATION == "3D":
                self._randomly_rotate_object(goal)
            r = (self.config.MAX_VIEW_DISTANCE - self.config.MIN_VIEW_DISTANCE) \
                    * np.random.random() + self.config.MIN_VIEW_DISTANCE
            a = 2 * np.pi * np.random.random()
            cand_pos = goal_pos + r * np.array([np.cos(a), 0, np.sin(a)])
            cand_rot = [0, np.sin(0.25 * np.pi - 0.5 * a), 0, np.cos(0.225 * np.pi - 0.5 * a)]
            if not self.config.OUT_OF_CONTEXT: # Need to check visibility in context
                island_r = self._sim.island_radius(cand_pos)
                filtered_step = self._sim.step_filter(cand_pos, goal_pos)
                if not (island_r > self.config.ISLAND_RADIUS 
                        and np.allclose(filtered_step, goal_pos)):
                    continue
            view = self._sim.get_observations_at(cand_pos, cand_rot)[self._sensor_uuid]
            goal._appearance_cache.append(view)

    def _move_object_out_of_context(self, goal: SpawnedObjectGoal):
        self._sim.set_translation(self.config.OUT_OF_CONTEXT_POS, goal._spawned_object_id)

    def _randomly_rotate_object(self, goal: SpawnedObjectGoal):
        rot_ax = np.random.random((3,))
        rot_ax /= np.linalg.norm(rot_ax)
        a = 2 * np.pi * np.random.random()
        rot = mn.Quaternion(np.sin(0.5 * a) * rot_ax, np.cos(0.5 * a))
        self._sim.set_rotation(rot, goal._spawned_object_id)

    def _restore_object_in_context(self, goal: SpawnedObjectGoal):
        self._sim.set_translation(goal.position, goal._spawned_object_id)
        if self.config.RANDOM_OBJECT_ORIENTATION == "3D":
            self._sim.set_rotation(mn.Quaternion(goal.orientation[:3], goal.orientation[3]),
                                   goal._spawned_object_id)

    def get_observation(self, episode: SpawnedObjectNavEpisode,
                        *args, **kwargs) -> VisualObservation:
        if any(goal._appearance_cache is None for goal in episode.goals):
            num_views_per_goal, more_views = divmod(self.config.NUM_VIEWS, len(episode.goals))
            for k, goal in enumerate(episode.goals):
                if self.config.OUT_OF_CONTEXT:
                    self._move_object_out_of_context(goal)
                num_views = num_views_per_goal + (1 if k < more_views else 0)
                self._generate_views_around_goal(goal, num_views)
                if self.config.OUT_OF_CONTEXT:
                    self._restore_object_in_context(goal)
            #TODO (gbono): what if some goal is barely visible from any points?
            # should we enforce balance of num_views between goals or not?
        views = np.array(list(chain.from_iterable(goal._appearance_cache
                                                  for goal in episode.goals)))
        np.random.shuffle(views)
        return views


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
        super().__init__(config, sim, dataset)

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
            self._sim.set_rotation(mn.Quaternion(goal.orientation[:3], goal.orientation[3]),
                                   goal._spawned_object_id)

    def reset(self, episode: SpawnedObjectNavEpisode) -> Observations:
        self._despawn_objects()
        self._reload_templates(episode)
        self._spawn_objects(episode)
        return super().reset(episode)
