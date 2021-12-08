import argparse
import threading
import io
import xmlrpc.server

import numpy as np
import gym

from SUBMISSION import SubmittedAgent


DEFAULT_RPC_PORT = 8000


def _deserialize_visual_observations(buf):
    buf = io.BytesIO(buf)
    return {uuid: obs for uuid, obs in np.load(buf).items()}


def run_rpc_agent(agent, port=DEFAULT_RPC_PORT):
    with xmlrpc.server.SimpleXMLRPCServer(("localhost", port),
                                          logRequests=False, allow_none=True) as rpc_server:
        @rpc_server.register_function
        def reset():
            agent.reset()

        @rpc_server.register_function
        def act(serialized_obs):
            obs = _deserialize_visual_observations(serialized_obs.data)
            return agent.act(obs)

        @rpc_server.register_function
        def terminate():
            t = threading.Thread(target=rpc_server.shutdown)
            t.start()

        print('agent server listening at', rpc_server.server_address, flush=True)
        print('AGENTSRVPORT', rpc_server.server_address[1], flush=True)
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
