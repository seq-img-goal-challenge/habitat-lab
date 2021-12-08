import argparse
import json

from habitat.utils.rpc_evaluation_client import run_rpc_benchmark, DEFAULT_RPC_PORT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c", default="configs/tasks/pointnav.yaml")
    parser.add_argument("--num-episodes", "-n", type=int, default=10)
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--num-retries", "-r", type=int, default=5)
    parser.add_argument("--retry-delay", "-d", type=float, default=1.0)
    return parser.parse_args()


def main(args):
    print('[eval container] evaluation started...', flush=True)
    metrics = run_rpc_benchmark(args.config_path, args.num_episodes, args.port)
    print("[eval container] METRICS >>>", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main(parse_args())
