import argparse

import numpy as np
import gym

from habitat.utils.rpc_evaluation_server import run_rpc_agent
from SUBMISSION import SubmittedAgent


DEFAULT_RPC_PORT = 8000


def _deserialize_visual_observations(buf):
    buf = io.BytesIO(buf)
    return {uuid: obs for uuid, obs in np.load(buf).items()}


def run_rpc_agent(agent, port=DEFAULT_RPC_PORT):
    with xmlrpc.server.SimpleXMLRPCServer(("localhost", port),
                                          logRequests=False,
                                          allow_none=True,
                                          use_builtin_types=True) as rpc_server:
        @rpc_server.register_function
        def check_init():
            return True

        @rpc_server.register_function
        def reset():
            agent.reset()

        @rpc_server.register_function
        def act(serialized_obs):
            obs = _deserialize_visual_observations(serialized_obs)
            return agent.act(obs)

        @rpc_server.register_function
        def terminate():
            t = threading.Thread(target=rpc_server.shutdown)
            t.start()

        rpc_server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_RPC_PORT)
    return parser.parse_args()


def main(args):
    agent = SubmittedAgent(gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (480, 640, 3),
                                                                  dtype=np.uint8)}),
                           gym.spaces.Discrete(4))
    run_rpc_agent(agent, args.port)


if __name__ == "__main__":
    main(parse_args())
