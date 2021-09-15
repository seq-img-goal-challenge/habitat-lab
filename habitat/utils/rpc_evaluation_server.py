from typing import Dict, Union, Any
import threading
import io
import xmlrpc.server

import numpy as np


DEFAULT_RPC_PORT = 8000


def _deserialize_visual_observations(buf: bytes) -> Dict[str, np.ndarray]:
    buf = io.BytesIO(buf)
    return {uuid: obs for uuid, obs in np.load(buf).items()}


def run_rpc_agent(agent: "habitat.Agent", port: int=DEFAULT_RPC_PORT) -> None:
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
