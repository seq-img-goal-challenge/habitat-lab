from typing import Optional, Dict, Union, Any
import io
import threading
import time
import xmlrpc.server
import xmlrpc.client

import numpy as np
import torch

from habitat.core.simulator import Observations
from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark


DEFAULT_RPC_PORT = 8000


def _serialize_visual_observations(observations: Observations) -> bytes:
    observations = {uuid: obs.numpy() if isinstance(obs, torch.Tensor) else obs
                    for uuid, obs in observations.items()}
    buf = io.BytesIO()
    np.savez(buf, **observations)
    return buf.getvalue()


def _deserialize_visual_observations(buf: bytes) -> Dict[str, np.ndarray]:
    buf = io.BytesIO(buf)
    return {uuid: obs for uuid, obs in np.load(buf).items()}


class AgentProxy(Agent):
    def __init__(self, agent_url: str, num_retries: int=5, retry_delay: float=1.0) -> None:
        self.proxy = xmlrpc.client.ServerProxy(agent_url, use_builtin_types=True)
        last_error = None
        for _ in range(num_retries):
            try:
                self.proxy.check_init()
                break
            except ConnectionRefusedError as e:
                last_error = e
                time.sleep(retry_delay)
        else:
            raise last_error

    def reset(self) -> None:
        self.proxy.reset()

    def act(self, observations: Observations) -> Union[int, str, Dict[str, Any]]:
        serialized_obs = _serialize_visual_observations(observations)
        return self.proxy.act(serialized_obs)

    def terminate(self) -> None:
        self.proxy.terminate()


def run_rpc_agent(agent: Agent, port: int=DEFAULT_RPC_PORT) -> None:
    with xmlrpc.server.SimpleXMLRPCServer(("localhost", port),
                                          logRequests=False,
                                          allow_none=True,
                                          use_builtin_types=True) as rpc_server:
        @rpc_server.register_function
        def check_init() -> bool:
            return True

        @rpc_server.register_function
        def reset() -> None:
            agent.reset()

        @rpc_server.register_function
        def act(serialized_obs: bytes) -> Union[int, str, Dict[str, Any]]:
            obs = _deserialize_visual_observations(serialized_obs)
            return agent.act(obs)

        @rpc_server.register_function
        def terminate() -> None:
            t = threading.Thread(target=rpc_server.shutdown)
            t.start()

        rpc_server.serve_forever()


def run_rpc_benchmark(config_paths: Optional[str]=None,
                      num_episodes: Optional[int]=None,
                      port: int=DEFAULT_RPC_PORT,
                      num_retries: int=5, retry_delay: float=1.0) -> Dict[str, float]:
    benchmark = Benchmark(config_paths, eval_remote=False)
    agent = AgentProxy(f"http://localhost:{port}", num_retries, retry_delay)
    metrics = benchmark.evaluate(agent, num_episodes)
    agent.terminate()
    return metrics
