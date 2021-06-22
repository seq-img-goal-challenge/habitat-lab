from typing import List, Dict, Set, Optional, Any
import argparse
import itertools
import os
import gzip

import numpy as np

import habitat
from habitat.core.simulator import Simulator
from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
        import create_object_pool, create_scene_pool, generate_spawned_objectgoal
from habitat.tasks.sequential_nav.sequential_objectnav import SequentialObjectNavStep, \
                                                              SequentialObjectNavEpisode


def generate_sequential_objectnav_episode(sim: Simulator,
                                          object_pool: Dict[str, Set[str]],
                                          category_index_map: Dict[str, int],
                                          ep_id: str,
                                          rng: np.random.Generator,
                                          min_seq_len: int,
                                          max_seq_len: int,
                                          max_goals: int,
                                          goal_radius: float,
                                          rotate_objects: str) -> SequentialObjectNavEpisode:
    start_pos = sim.sample_navigable_point()
    a = 2 * np.pi * rng.random()
    start_rot = [*(np.sin(0.5 * a) * sim.up_vector), np.cos(0.5 * a)]

    step_count = rng.integers(min_seq_len, max_seq_len, endpoint=True)
    selected_categories = rng.choice(list(object_pool.items()), step_count, replace=False)
    steps = []
    for category, tmpl_ids in selected_categories:
        cat_index = category_index_map[category]
        if len(tmpl_ids) > max_goals:
            tmpl_ids = rng.choice(list(tmpl_ids), max_goals, replace=True)
        goals = [generate_spawned_objectgoal(sim, start_pos, tmpl_id, rng,
                                             max_goals, goal_radius, rotate_objects)
                 for tmpl_id in tmpl_ids]
        steps.append(SequentialObjectNavStep(object_category=category,
                                             object_category_index=cat_index,
                                             goals=goals))

    return SequentialObjectNavEpisode(episode_id=ep_id,
                                      scene_id=sim.habitat_config.SCENE,
                                      start_position=start_pos,
                                      start_rotation=start_rot,
                                      steps=steps)


def generate_sequential_objectnav_dataset(config_path: str, extra_config: List[str],
                                          num_episodes:int, min_seq_len: int, max_seq_len: int,
                                          max_goals: int, goal_radius: float,
                                          rotate_objects: str, if_exist: str,
                                          scenes_dir: str, objects_dir: str,
                                          seed: Optional[int]=None) -> None:
    cfg = habitat.get_config(config_path, extra_config)
    out_path = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)

    try:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
        if if_exist == "ABORT":
            print("'{}' already exists, aborting".format(out_path))
            sys.exit()
        elif if_exist == "OVERRIDE":
            dataset.episodes = []
    except FileNotFoundError:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE)
    new_episodes = []
    ep_id = (str(i) for i in itertools.count())

    rng = np.random.default_rng(seed)
    scene_pool = create_scene_pool(scenes_dir)
    object_pool, cat_idx_map = create_object_pool(objects_dir)

    num_ep_per_scene, more_ep = divmod(num_episodes, len(scene_pool))
    for k, scene in enumerate(scene_pool):
        if num_ep_per_scene == 0 and k >= more_ep:
            break
        cfg.SIMULATOR.defrost()
        cfg.SIMULATOR.SCENE = scene
        cfg.freeze()
        with habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
            if seed is not None:
                sim.seed(seed + k)
            for _ in range(num_ep_per_scene + (1 if k < more_ep else 0)):
                episode = generate_sequential_objectnav_episode(sim, object_pool, cat_idx_map, 
                                                                next(ep_id), rng,
                                                                min_seq_len, max_seq_len,
                                                                max_goals, goal_radius,
                                                                rotate_objects)
                new_episodes.append(episode)
    dataset.episodes.extend(new_episodes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, 'wt') as f:
        f.write(dataset.to_json())


_DEFAULT_ARGS: Dict[str, Any] = {"config_path": "configs/tasks/pointnav_gibson.yaml",
                                 "seed": None,
                                 "num_episodes": 4000,
                                 "min_seq_len": 3,
                                 "max_seq_len": 8,
                                 "max_goals": 5,
                                 "goal_radius": 1.0,
                                 "rotate_objects": "DISABLE",
                                 "if_exist": "ABORT",
                                 "scenes_dir": "data/scene_datasets/gibson",
                                 "objects_dir": "data/object_datasets/test_objects"}


def _parse_args(argv: Optional[List[str]]=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--seed", "-s", type=int)
    parser.add_argument("--num-episodes", "-n", type=int)
    parser.add_argument("--min-seq-len", "-k", type=int)
    parser.add_argument("--max-seq-len", "-l", type=int)
    parser.add_argument("--max-goals", "-m", type=int)
    parser.add_argument("--goal-radius", "-r", type=float)
    parser.add_argument("--rotate-objects", choices=("DISABLE", "YAXIS", "3D"))
    parser.add_argument("--if-exist", choices=("ABORT", "OVERRIDE", "APPEND"))
    parser.add_argument("--scenes-dir")
    parser.add_argument("--objects-dir")
    parser.add_argument("extra_config", nargs=argparse.REMAINDER)
    parser.set_defaults(**_DEFAULT_ARGS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate_sequential_objectnav_dataset(**vars(_parse_args()))
