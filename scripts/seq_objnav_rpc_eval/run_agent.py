import argparse

import numpy as np
import gym

from habitat.utils.rpc_evaluation_server import run_rpc_agent
from SUBMISSION.agent import SubmittedAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int)
    return parser.parse_args()


def main(args):
    agent = SubmittedAgent(gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, (480, 640, 3),
                                                                  dtype=np.uint8)}),
                           gym.spaces.Discrete(4))
    run_rpc_agent(agent, args.port)


if __name__ == "__main__":
    main(parse_args())
