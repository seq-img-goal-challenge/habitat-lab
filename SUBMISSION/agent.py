from typing import Dict, Union, Any
import numpy as np
import gym


class SubmittedAgent:
    """
    Fill in the methods of this class!
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        """
        Called once at the beginning of the evaluation.
        Setup your NN, load weights, etc...
        """
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self) -> None:
        """
        Called at the beginning of every episode.
        Reset the hidden states of your RNNs, etc...
        """
        pass

    def act(self, observations: Dict[str, np.ndarray]) -> Union[int, str, Dict[str, Any]]:
        """
        Called at every step in an episode.
        Observations are not batched, don't forget to prepend a dim of size 1 to the arrays.
        Preferred action format is int (for RPC serialization),
        but str (action name) and dict ({"action": action index}) are also supported
        (cf. habitat.Env).
        """
        return self.action_space.sample()
