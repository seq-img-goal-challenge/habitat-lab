import argparse
import json

from habitat.utils.local_rpc_evaluation import run_rpc_benchmark, DEFAULT_RPC_PORT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c", default="configs/tasks/pointnav.yaml")
    parser.add_argument("--num-episodes", "-n", default=10)
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_RPC_PORT)
    return parser.parse_args()


def main(args):
    print('[eval container] evaluation started...', flush=True)
    metrics = run_rpc_benchmark(args.config_path, args.num_episodes, args.port)
    print("[eval container] METRICS >>>", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main(parse_args())
