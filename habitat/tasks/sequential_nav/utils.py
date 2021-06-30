from typing import Any, List, Optional
from collections import OrderedDict

import numpy as np
from gym import Space, spaces

from habitat.config.default import Config
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode


def _extend_space_shape_recursive(space: Space, seq_len: int, pad_val: Any) -> Space:
    if isinstance(space, spaces.Discrete):
        return spaces.MultiDiscrete([space.n + 1 for _ in range(seq_len)])
    elif isinstance(space, spaces.MultiDiscrete):
        extended_shape = (seq_len,) + space.shape
        return spaces.MultiDiscrete(np.broadcast_to(space.nvec + 1, extended_shape))
    elif isinstance(space, spaces.MultiBinary):
        return spaces.MultiDiscrete(np.full((seq_len,) + space.shape, 2, dtype=np.uint8))
    elif isinstance(space, spaces.Box):
        extended_shape = (seq_len,) + space.shape
        return spaces.Box(low=np.minimum(np.broadcast_to(space.low, extended_shape), pad_val),
                          high=np.maximum(np.broadcast_to(space.high, extended_shape), pad_val),
                          dtype=space.dtype)
    elif isinstance(space, spaces.Tuple):
        return spaces.Tuple(tuple(_extend_space_shape_recursive(sub_space, seq_len, pad_val)
                                  for sub_space in space.spaces))
    elif isinstance(space, spaces.Dict):
        return spaces.Dict([(k, _extend_space_shape_recursive(sub_space, seq_len, pad_val))
                            for k, sub_space in space.spaces.items()])
    else:
        raise RuntimeError("Observation spaces of type '{}' ".format(type(space)) \
                           + "cannot be extend for sequences of observations.")


def _pack_obs_seq_recursive(obs_seq: List[Any], seq_len: int, pad_val: Any) -> Any:
    peek = obs_seq[0]
    pad_len = seq_len - len(obs_seq)
    #TODO(gbono): pad_val properly works only for Box...
    if isinstance(peek, int): # Discrete
        return np.array(obs_seq + [pad_val for _ in range(pad_len)])
    if isinstance(peek, np.ndarray): # MultiDiscrete, MultiBinary, Box
        obs_seq = np.stack(obs_seq, 0)
        pad = np.full((pad_len,) + peek.shape, pad_val)
        return np.concatenate((obs_seq, pad), 0)
    elif isinstance(peek, tuple): # Tuple
        return tuple(_pack_obs_seq_recursive(list(sub_seq), seq_len, pad_val)
                     for sub_seq in zip(*obs_seq))
    elif isinstance(peek, OrderedDict): # Dict
        return OrderedDict((k, _pack_obs_seq_recursive([obs[k] for obs in obs_seq],
                                                       seq_len, pad_val)) for k in peek.keys())
    else:
        raise RuntimeError("Observations of type '{}' ".format(type(step_obs[0])) \
                           + "cannot be packed as sequences.")


def make_sequential(base_sensor_cls, *, name=None):
    if name is None:
        name = "Sequential" + base_sensor_cls.__name__

    @registry.register_sensor(name=name)
    class SequentialSensor(base_sensor_cls):
        _max_seq_len: int
        _pad_val: int
        _seq_mode: str
        _prv_step_idx: Optional[int]

        def __init__(self, *args: Any, sim: Simulator, config: Config,
                     dataset: "SequentialDataset", **kwargs: Any) -> None:
            self._max_seq_len = dataset.get_max_sequence_len()
            self._pad_val = config.PADDING_VALUE
            self._seq_mode = config.SEQUENTIAL_MODE
            self._prv_step_idx = None
            super().__init__(*args, sim=sim, config=config, dataset=dataset, **kwargs)

        def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
            src_space = super()._get_observation_space(*args, **kwargs)
            if self.config.SEQUENTIAL_MODE == "MYOPIC":
                return src_space
            elif isinstance(src_space, spaces.Box):
                extended_shape = (self._max_seq_len,) + src_space.shape
                low = np.broadcast_to(src_space.low, extended_shape)
                high = np.broadcast_to(src_space.high, extended_shape)
                return spaces.Box(low=np.minimum(low, self._pad_val),
                                  high=np.maximum(high, self._pad_val),
                                  dtype=src_space.dtype)
            else:
                raise NotImplementedError(f"Cannot make sequential sensor" \
                        + "for observation space of type '{type(src_space)}'")

        def _get_observation_for_step(self, episode: "SequentialEpisode", step_index: int,
                                      *args: Any, **kwargs: Any) -> Any:
            step = episode.steps[step_index]
            step_id = f"{episode.episode_id}_{step_index}"
            step_as_episode = NavigationEpisode(episode_id=step_id,
                                                scene_id=episode.scene_id,
                                                start_position=episode.start_position,
                                                start_rotation=episode.start_rotation,
                                                goals=step.goals)
            for k, v in vars(episode).items():
                if k not in vars(step_as_episode):
                    setattr(step_as_episode, k, v)
            for k, v in vars(step).items():
                if k not in vars(step_as_episode):
                    setattr(step_as_episode, k, v)
            return super().get_observation(*args, episode=step_as_episode, **kwargs)

        def get_observation(self, *args: Any, episode: "SequentialEpisode",
                            **kwargs: Any) -> Any:
            step_idx = episode._current_step_index
            if step_idx < 0 or step_idx >= episode.num_steps:
                if self._prv_step_idx is None:
                    raise ValueError("Invalid step index encountered" \
                                     + "while getting the observation of a sequential sensor.")
                else:
                    step_idx = self._prv_step_idx
            self._prv_step_idx = step_idx

            if self._seq_mode == "MYOPIC":
                return self._get_observation_for_step(episode, step_idx, *args, **kwargs)

            if self._seq_mode == "PREFIX":
                indices = range(0, step_idx + 1)
            elif self._seq_mode == "SUFFIX":
                indices = range(step_idx, episode.num_steps)
            elif self._seq_mode == "FULL":
                indices = range(0, episode.num_steps)
            obs_seq = np.stack([self._get_observation_for_step(episode, idx, *args, **kwargs)
                                for idx in indices], 0)
            pad_len = self._max_seq_len - obs_seq.shape[0]
            pad = np.full((pad_len,) + obs_seq.shape[1:], self._pad_val)
            return np.concatenate([obs_seq, pad], 0)

    return SequentialSensor
