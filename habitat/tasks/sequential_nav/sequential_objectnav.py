from typing import List

import attr

from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, \
                                                SpawnedObjectGoalCategorySensor, \
                                                SpawnedObjectGoalAppearanceSensor, \
                                                SpawnedObjectNavTask
from habitat.tasks.sequential_nav.sequential_nav import SequentialStep, \
                                                        SequentialEpisode, \
                                                        SequentialNavigationTask, \
                                                        DistanceToNextGoal
from habitat.tasks.sequential_nav.utils import make_sequential


@attr.s(auto_attribs=True, kw_only=True)
class SequentialObjectNavStep(SequentialStep):
    goals: List[SpawnedObjectGoal] = attr.ib(default=None, validator=not_none_validator)
    object_category: str
    object_category_index: int


@attr.s(auto_attribs=True, kw_only=True)
class SequentialObjectNavEpisode(SequentialEpisode):
    steps: List[SequentialObjectNavStep] = attr.ib(default=None, validator=not_none_validator)


@registry.register_task(name="SequentialObjectNav-v0")
class SequentialObjectNavTask(SpawnedObjectNavTask, SequentialNavigationTask):
    def _spawn_objects(self, episode: SequentialObjectNavEpisode) -> None:
        for step in episode.steps:
            super()._spawn_objects(step)


SequentialObjectGoalCategorySensor = make_sequential(SpawnedObjectGoalCategorySensor,
                                                     name="SequentialObjectGoalCategorySensor")
SequentialObjectGoalAppearanceSensor = make_sequential(SpawnedObjectGoalAppearanceSensor,
                                                       name="SequentialObjectGoalAppearanceSensor")

@registry.register_measure
class DistanceToNextObject(DistanceToNextGoal):
    _step_targets: List[List[float]]

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        super().__init__(*args, sim=sim, **kwargs)
        self._step_targets = []

    def _get_targets_for_goal(self, goal: SpawnedObjectGoal):
        bb = self._sim.get_object_scene_node(goal._spawned_object_id).cumulative_bb
        shifts = np.array([[x, 0, z] for x in (0, bb.left, bb.right)
                                     for z in (0, bb.back, bb.front)])
        targets = np.array(goal.position)[None, :] + shifts
        return targets.tolist()

    def _update_step_targets(self, episode: SequentialObjectNavEpisode) -> None:
        idx = min(episode._current_step_index, episode.num_steps - 1)
        self._step_targets = []
        for goal in episode.steps[idx].goals:
            self._step_targets.extend(self._get_targets_for_goal(goal))

    def _compute_distance(self, pos: np.ndarray,
                                episode: SequentialObjectNavEpisode) -> float:
        if not self._step_targets or self._last_step_index != episode._current_step_index:
            self._update_step_targets(episode)
        return self._sim.geodesic_distance(pos, self._step_targets)
