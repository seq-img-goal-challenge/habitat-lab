from typing import List, Any

import attr
import numpy as np

from habitat.core.registry import registry
from habitat.core.simulator import Simulator
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
    def _spawn_objects(self) -> None:
        for step in self._current_episode.steps:
            for goal in step.goals:
                goal._spawn_in_sim(self._sim)

    def _despawn_objects(self) -> None:
        for step in self._current_episode.steps:
            for goal in step.goals:
                goal._despawn_from_sim(self._sim)


SequentialObjectGoalCategorySensor = make_sequential(SpawnedObjectGoalCategorySensor,
                                                     name="SequentialObjectGoalCategorySensor")
SequentialObjectGoalAppearanceSensor = make_sequential(SpawnedObjectGoalAppearanceSensor,
                                                       name="SequentialObjectGoalAppearanceSensor")
