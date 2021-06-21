from typing import Any, Optional, Tuple, List, Set, Dict
import argparse
import sys
import os
import collections
import itertools
import gzip

import numpy as np

import habitat
from habitat.core.simulator import Simulator
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode
from habitat.datasets.spawned_objectnav.utils import DEFAULT_SCENE_PATH_PREFIX, \
                                                     DEFAULT_SCENE_PATH_EXT, \
                                                     DEFAULT_OBJECT_PATH_PREFIX, \
                                                     DEFAULT_OBJECT_PATH_EXT


def create_object_pool(objects_dir: str) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    cat_idx = itertools.count()
    cat_idx_map = {}
    if not os.path.isdir(objects_dir):
        objects_dir = os.path.join(DEFAULT_OBJECT_PATH_PREFIX, objects_dir)
    pool = collections.defaultdict(set)
    for entry in os.scandir(objects_dir):
        if entry.is_dir():
            category = entry.name
            if category not in cat_idx_map:
                cat_idx_map[category] = next(cat_idx)
            for sub_entry in os.scandir(entry.path):
                if sub_entry.is_file() and sub_entry.name.endswith(DEFAULT_OBJECT_PATH_EXT):
                    tmpl_id = sub_entry.path
                    pool[category].add(tmpl_id)
        elif entry.is_file() and entry.name.endswith(DEFAULT_OBJECT_PATH_EXT):
            category = entry.name[:-len(DEFAULT_OBJECT_PATH_EXT)]
            if category not in cat_idx_map:
                cat_idx_map[category] = next(cat_idx)
            tmpl_id = entry.path
            pool[category].add(tmpl_id)
    return pool, cat_idx_map


def create_scene_pool(scenes_dir: str) -> Set[str]:
    pool = set()
    if not os.path.isdir(scenes_dir):
        scenes_dir = os.path.join(DEFAULT_SCENE_PATH_PREFIX, scenes_dir)
    for entry in os.scandir(scenes_dir):
        if entry.is_file() and entry.name.endswith(DEFAULT_SCENE_PATH_EXT):
            pool.add(entry.path)
    return pool


def generate_spawned_objectgoal(sim: Simulator,
                                start_pos: np.ndarray,
                                tmpl_id: str,
                                rng: np.random.Generator,
                                max_goals: int,
                                goal_radius: float,
                                rotate_objects: str) -> SpawnedObjectGoal:
    d = np.inf
    while not np.isfinite(d):
        obj_pos = sim.sample_navigable_point()
        d = sim.geodesic_distance(start_pos, obj_pos)
    if rotate_objects == "DISABLE":
        obj_rot = [0.0, 0.0, 0.0, 1.0]
    elif rotate_objects == "YAXIS":
        a = 2 * np.pi * rng.random()
        obj_rot =  [*(np.sin(0.5 * a) * sim.up_vector), np.cos(0.5 * a)]
    elif rotate_objects == "3D":
        rot_ax = rng.random((3,))
        rot_ax /= np.linalg.norm(rot_ax)
        a = 2 * np.pi * rng.random()
        obj_rot = [*(np.sin(0.5 * a) * rot_ax), np.cos(0.5 * a)]
    return SpawnedObjectGoal(position=obj_pos,
                             orientation=obj_rot,
                             radius=goal_radius,
                             object_template_id=tmpl_id)


def generate_spawned_objectnav_episode(sim: Simulator,
                                       object_pool: Dict[str, Set[str]],
                                       category_index_map: Dict[str, int],
                                       ep_id: str,
                                       rng: np.random.Generator,
                                       max_goals: int,
                                       goal_radius: float,
                                       rotate_objects: str) -> SpawnedObjectNavEpisode:
    start_pos = sim.sample_navigable_point()
    a = 2 * np.pi * rng.random()
    start_rot = [*(np.sin(0.5 * a) * sim.up_vector), np.cos(0.5 * a)]

    category, tmpl_ids = rng.choice(list(object_pool.items()))
    cat_index = category_index_map[category]
    if len(tmpl_ids) > max_goals:
        tmpl_ids = rng.choice(list(tmpl_ids), max_goals, replace=True)
    goals = [generate_spawned_objectgoal(sim, start_pos, tmpl_id, rng,
                                         max_goals, goal_radius, rotate_objects)
             for tmpl_id in tmpl_ids]

    return SpawnedObjectNavEpisode(episode_id=ep_id,
                                   scene_id=sim.habitat_config.SCENE,
                                   start_position=start_pos,
                                   start_rotation=start_rot,
                                   object_category=category,
                                   object_category_index=cat_index,
                                   goals=goals)


def generate_spawned_objectnav_dataset(config_path: str, extra_config: List[str],
                                       num_episodes:int, max_goals: int, goal_radius: float,
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
                episode = generate_spawned_objectnav_episode(sim, object_pool, cat_idx_map, 
                                                             next(ep_id), rng,
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
    generate_spawned_objectnav_dataset(**vars(_parse_args()))
