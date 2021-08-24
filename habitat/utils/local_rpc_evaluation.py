from typing import Optional, Dict, Union, Any
import io
import threading
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy, Binary

import numpy as np
import torch

from habitat.core.simulator import Observations
from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark


RPC_SERVER_PORT = 8000


def _serialize_visual_observations(observations: Observations) -> bytes:
    observations = {uuid: obs.numpy() if isinstance(obs, torch.Tensor) else obs
                    for uuid, obs in observations.items()}
    buf = io.BytesIO()
    np.savez(buf, **observations)
    return buf.getvalue()


def _deserialize_visual_observations(buf: bytes) -> Dict[str, np.ndarray]:
    buf = io.BytesIO(buf)
    return np.load(buf)


def run_rpc_agent(agent: Agent) -> None:
    with SimpleXMLRPCServer(("localhost", RPC_SERVER_PORT),
                            logRequests=False, allow_none=True) as rpc_server:
        @rpc_server.register_function
        def reset() -> None:
            agent.reset()

        @rpc_server.register_function
        def act(serialized_obs: Binary) -> Union[int, str, Dict[str, Any]]:
            obs = _deserialize_visual_observations(serialized_obs.data)
            return agent.act(obs)

        @rpc_server.register_function
        def terminate() -> None:
            t = threading.Thread(target=rpc_server.shutdown)
            t.start()

        rpc_server.serve_forever()


class AgentProxy(Agent):
    def __init__(self, agent_url) -> None:
        self.proxy = ServerProxy(agent_url)

    def reset(self):
        self.proxy.reset()

    def act(self, observations: Observations) -> Union[int, str, Dict[str, Any]]:
        serialized_obs = _serialize_visual_observations(observations)
        return self.proxy.act(serialized_obs)

    def terminate(self):
        self.proxy.terminate()


def run_rpc_benchmark(config_paths: Optional[str]=None,
                      agent_url: Optional[str]=None,
                      num_episodes: Optional[int]=None) -> Dict[str, float]:
    benchmark = Benchmark(config_paths, False)
    if agent_url is None:
        agent_url = f"http://localhost:{RPC_SERVER_PORT}"
    agent = AgentProxy(agent_url)
    metrics = benchmark.evaluate(agent, num_episodes)
    agent.terminate()
    return metrics
