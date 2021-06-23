from typing import Any, Optional, List

import attr
import numpy as np
from gym import Space, spaces

from habitat.config.default import Config
from habitat.core.embodied_task import Measure, Action
from habitat.core.dataset import Episode
from habitat.core.simulator import Simulator, Observations
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, \
                                                SpawnedObjectGoalCategorySensor, \
                                                SpawnedObjectGoalAppearanceSensor, \
                                                SpawnedObjectNavTask
from habitat.tasks.sequential_nav.sequential_nav import SequentialStep, \
                                                        SequentialEpisode, \
                                                        SequentialNavigationTask


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


@registry.register_sensor
class SequentialObjectGoalCategorySensor(SpawnedObjectGoalCategorySensor):
    _max_seq_len: int
    _pad_val: int

    def __init__(self, *args: Any, config: Config, dataset: "SequentialObjectNavDatasetV1",
                 **kwargs: Any) -> None:
        self._max_seq_len = dataset.get_max_sequence_len()
        self._pad_val = config.PADDING_VALUE
        super().__init__(config=config, dataset=dataset, *args, **kwargs)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        if self.config.SEQUENTIAL_MODE == "MYOPIC":
            return super()._get_observation_space(*args, **kwargs)
        else:
            return spaces.Box(low=min(0, self._pad_val),
                              high=max(self._max_object_category_index, self._pad_val),
                              shape=(self._max_seq_len, 1), dtype=np.int64)

    def get_observation(self, *args: Any, episode: SequentialObjectNavEpisode,
                        **kwargs: Any) -> np.ndarray:
        if self.config.SEQUENTIAL_MODE == "MYOPIC":
            return np.array([episode.steps[episode._current_step_index].object_category_index],
                            dtype=np.int64)

        if self.config.SEQUENTIAL_MODE == "SUFFIX":
            steps = episode.steps[episode._current_step_index:]
        elif self.config.SEQUENTIAL_MODE == "FULL":
            steps = episode.steps
        return np.array([step.object_category_index for step in steps], dtype=np.int64)


@registry.register_sensor
class SequentialObjectGoalAppearanceSensor(SpawnedObjectGoalAppearanceSensor):
    _max_seq_len: int
    _pad_val: int

    def __init__(self, *args: Any, config: Config, sim: Simulator,
                 dataset: "SequentialObjectNavDatasetV1", **kwargs: Any) -> None:
        self._max_seq_len = dataset.get_max_sequence_len()
        self._pad_val = config.PADDING_VALUE
        super().__init__(config=config, sim=sim, *args, **kwargs)

    def _get_observation_space(self, *args:Any, **kwargs: Any) -> Space:
        src_space = super()._get_observation_space(*args, **kwargs)
        if self.config.SEQUENTIAL_MODE == "MYOPIC":
            return src_space
        else:
            extended_shape = (self._max_seq_len,) + src_space.shape
            low = np.broadcast_to(src_space.low, extended_shape)
            high = np.broadcast_to(src_space.high, extended_shape)
            return spaces.Box(low=np.min(low, self._pad_val),
                              high=np.max(high, self._pad_val),
                              dtype=src_space.dtype)

    def get_observation(self, *args: Any, episode: SequentialObjectNavEpisode,
                        **kwargs: Any) -> np.ndarray:
        if self.config.SEQUENTIAL_MODE == "MYOPIC":
            return super().get_observation(episode.steps[episode._current_step_index])

        if self.config.SEQUENTIAL_MODE == "SUFFIX":
            steps = episode.steps[episode._current_step_index:]
        elif self.config.SEQUENTIAL_MODE == "FULL":
            steps = episode.steps
        pad_len = self._max_seq_len - len(steps)
        obs = np.stack([super().get_observation(step) for step in steps], 0)
        pad = np.full(self._pad_val, (pad_len,) + obs.shape[1:])
        return np.concatenate([obs, pad], 0)
