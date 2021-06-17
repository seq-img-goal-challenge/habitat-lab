from typing import Any, List
from collections import OrderedDict

import numpy as np
from gym import Space, spaces


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

