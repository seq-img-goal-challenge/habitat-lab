from typing import List, Dict, Set, Optional, Any
import argparse
import itertools
import os
import gzip

import numpy as np
import tqdm

import habitat
from habitat.core.simulator import Simulator
from habitat.datasets.spawned_objectnav.spawned_objectnav_generator \
        import ObjectPoolCategory, ObjectRotation, ExistBehavior, \
               UnreachableGoalError, MaxRetriesError, \
               create_object_pool, create_scene_pool, generate_spawned_objectgoals
from habitat.tasks.sequential_nav.sequential_objectnav import SequentialObjectNavStep, \
                                                              SequentialObjectNavEpisode


def generate_sequential_objectnav_episode(sim: Simulator,
                                          ep_id: str,
                                          min_seq_len: int,
                                          max_seq_len: int,
                                          max_goals: int,
                                          object_pool: List[ObjectPoolCategory],
                                          rotate_objects: ObjectRotation,
                                          num_retries: int,
                                          rng: np.random.Generator) \
                                         -> SequentialObjectNavEpisode:
    start_pos = sim.sample_navigable_point()
    a = 2 * np.pi * rng.random()
    start_rot = [*(np.sin(0.5 * a) * sim.up_vector), np.cos(0.5 * a)]

    step_count = rng.integers(min_seq_len, max_seq_len, endpoint=True)
    selected_categories = rng.choice(object_pool, step_count, replace=False)
    steps = []
    all_tmpl_ids = []
    step_splits = []
    for category, cat_index, tmpl_ids in selected_categories:
        prv_len = len(all_tmpl_ids)
        if len(tmpl_ids) > max_goals:
            all_tmpl_ids.extend(rng.choice(tmpl_ids, max_goals, replace=True))
        else:
            all_tmpl_ids.extend(tmpl_ids)
        nxt_len = len(all_tmpl_ids)
        step_splits.append((category, cat_index, prv_len, nxt_len))
        prv_len = nxt_len

    errors = []
    for retry in range(num_retries):
        try:
            goals = generate_spawned_objectgoals(sim, start_pos, all_tmpl_ids,
                                                 rotate_objects, rng)
            break
        except UnreachableGoalError as e:
            errors.append(e)
    else:
        raise MaxRetriesError("generate reachable goals", num_retries, errors)
    steps = [SequentialObjectNavStep(object_category=category,
                                     object_category_index=cat_index,
                                     goals=goals[first:last])
             for category, cat_index, first, last in step_splits]
    return SequentialObjectNavEpisode(episode_id=ep_id,
                                      scene_id=sim.habitat_config.SCENE,
                                      start_position=start_pos,
                                      start_rotation=start_rot,
                                      steps=steps)


def generate_sequential_objectnav_dataset(config_path: str, extra_config: List[str],
                                          scenes_dir: str, objects_dir: str,
                                          num_episodes:int,
                                          min_seq_len: int, max_seq_len: int, max_goals: int,
                                          rotate_objects: ObjectRotation,
                                          if_exist: ExistBehavior,
                                          num_retries: int,
                                          seed: Optional[int]=None) -> None:
    cfg = habitat.get_config(config_path, extra_config)
    out_path = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)

    try:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
        if if_exist is ExistBehavior.ABORT:
            print("'{}' already exists, aborting".format(out_path))
            sys.exit()
        elif if_exist is ExistBehavior.OVERRIDE:
            dataset.episodes = []
        elif if_exist is ExistBehavior.APPEND:
            pass
    except FileNotFoundError:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE)
    new_episodes = []
    ep_id = (str(i) for i in itertools.count())

    rng = np.random.default_rng(seed)
    scene_pool = create_scene_pool(scenes_dir)
    object_pool = create_object_pool(objects_dir)

    num_ep_per_scene, more_ep = divmod(num_episodes, len(scene_pool))
    with tqdm.tqdm(total=num_episodes) as progress:
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
                    episode = generate_sequential_objectnav_episode(sim, next(ep_id),
                                                                    min_seq_len, max_seq_len,
                                                                    max_goals,
                                                                    object_pool,
                                                                    rotate_objects,
                                                                    num_retries,
                                                                    rng)
                    new_episodes.append(episode)
                    progress.update()
    dataset.episodes.extend(new_episodes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, 'wt') as f:
        f.write(dataset.to_json())


_DEFAULT_ARGS: Dict[str, Any] = {"config_path": "configs/tasks/pointnav_gibson.yaml",
                                 "scenes_dir": "data/scene_datasets/gibson",
                                 "objects_dir": "data/object_datasets/test_objects",
                                 "num_episodes": 25,
                                 "min_seq_len": 1,
                                 "max_seq_len": 1,
                                 "max_goals": 1,
                                 "rotate_objects": ObjectRotation.FIXED,
                                 "if_exist": ExistBehavior.ABORT,
                                 "num_retries": 4,
                                 "seed": None}


def _parse_args(argv: Optional[List[str]]=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--scenes-dir")
    parser.add_argument("--objects-dir")
    parser.add_argument("--num-episodes", "-n", type=int)
    parser.add_argument("--min-seq-len", "-k", type=int)
    parser.add_argument("--max-seq-len", "-l", type=int)
    parser.add_argument("--max-goals", "-m", type=int)
    parser.add_argument("--rotate-objects", type=lambda name: ObjectRotation[name],
                        choices=list(ObjectRotation))
    parser.add_argument("--if-exist", type=lambda name: ExistBehavior[name],
                        choices=list(ExistBehavior))
    parser.add_argument("--num-retries", "-r", type=int)
    parser.add_argument("--seed", "-s", type=int)
    parser.add_argument("extra_config", nargs=argparse.REMAINDER)
    parser.set_defaults(**_DEFAULT_ARGS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate_sequential_objectnav_dataset(**vars(_parse_args()))
