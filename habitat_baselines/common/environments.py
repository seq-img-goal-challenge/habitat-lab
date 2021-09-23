#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type, Tuple, Any

import numpy as np

import habitat
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="RearrangeRLEnv")
class RearrangeRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="SequentialNavRLEnv")
class SequentialNavRLEnv(NavRLEnv):
    _seq_reward_measure_name: str
    _prv_seq_measure: Any
    _slack_r: float
    _success_r: float

    def __init__(self, config: Config, dataset: Optional[Dataset]=None) -> None:
        self._seq_reward_measure_name = config.RL.SEQUENTIAL_REWARD_MEASURE
        self._prv_seq_measure = None
        self._slack_r = config.RL.SLACK_REWARD
        self._success_r = config.RL.SUCCESS_REWARD
        super().__init__(config, dataset)

    def reset(self):
        observations = super().reset()
        self._prv_seq_measure = self._env.get_metrics()[self._seq_reward_measure_name]
        return observations

    def get_reward_range(self) -> Tuple[float, float]:
        max_distance_delta = self._core_env_config.SIMULATOR.FORWARD_STEP_SIZE
        max_progress_delta = 1.0
        return (self._slack_r - max_distance_delta,
                self._slack_r + self._success_r + max_distance_delta + max_progress_delta)

    def get_reward(self, observations: Observations) -> float:
        reward = self._rl_config.SLACK_REWARD
        m = self._env.get_metrics()

        cur_measure = m[self._reward_measure_name]
        cur_seq_measure = m[self._seq_reward_measure_name]

        progress_delta = cur_seq_measure - self._prv_seq_measure
        if progress_delta > 0: # Progress can only increase
            reward += self._rl_config.PROGRESS_REWARD
            self._prv_seq_measure = cur_seq_measure
            # Reset reward measure if some progress was made
            # (otherwise would penalize going to next step...)
            self._previous_measure = cur_measure

        # Assume reward measure is similar to a distance to goal
        # hence, encourage a decrease in measure and penalize an increase
        measure_delta = cur_measure - self._previous_measure
        reward -= measure_delta
        self._previous_measure = cur_measure

        # Add a sparse success reward
        success = m[self._success_measure_name]
        if success:
            reward += self._rl_config.SUCCESS_REWARD
        return reward

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over
