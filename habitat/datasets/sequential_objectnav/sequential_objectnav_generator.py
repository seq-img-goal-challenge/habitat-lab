from typing import List, Dict, Set, Optional, Any
import argparse
import os
import gzip
import itertools
import logging

import numpy as np
import tqdm

import habitat
from habitat.config.default import Config
from habitat.core.simulator import Simulator
from habitat.datasets.spawned_objectnav.spawned_objectnav_generator import (
    DEFAULT_SCENE_PATH_EXT, ObjectPoolCategory, ObjectRotation, ExistBehavior, \
    UnreachableGoalError, MaxRetriesError, create_object_pool_v2, create_scene_pool,
    generate_spawned_objectgoals, check_existence
)
from habitat.datasets.sequential_objectnav.sequential_objectnav_dataset \
        import SequentialObjectNavDatasetV0
from habitat.tasks.sequential_nav.sequential_objectnav import SequentialObjectNavStep, \
                                                              SequentialObjectNavEpisode


_filename, _ = os.path.splitext(os.path.basename(__file__))
_logger = habitat.logger.getChild(_filename)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[{asctime}] {levelname} ({name}): {msg}", style='{'))
_logger.addHandler(_handler)


def generate_sequential_objectnav_episode(sim: Simulator,
                                          ep_id: str,
                                          min_seq_len: int,
                                          max_seq_len: int,
                                          max_goals: int,
                                          object_pool: List[ObjectPoolCategory],
                                          rotate_objects: ObjectRotation,
                                          view_pts_cfg: Config,
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
                                                 rotate_objects, view_pts_cfg, rng)
            break
        except UnreachableGoalError as e:
            errors.append(e)
    else:
        short_scene, _ = os.path.splitext(os.path.basename(sim.habitat_config.SCENE))
        raise MaxRetriesError(f"generate reachable goals in scene '{short_scene}'",
                              num_retries, errors)
    steps = [SequentialObjectNavStep(object_category=category,
                                     object_category_index=cat_index,
                                     goals=goals[first:last])
             for category, cat_index, first, last in step_splits]
    episode = SequentialObjectNavEpisode(episode_id=ep_id,
                                         scene_id=sim.habitat_config.SCENE,
                                         start_position=start_pos,
                                         start_rotation=start_rot,
                                         steps=steps)
    _logger.info(f"Successfully generated episode '{ep_id}'.")
    return episode


def generate_sequential_objectnav_dataset(cfg: Config, scenes_dir: str, objects_dir: str,
                                          num_episodes:int, min_seq_len: int, max_seq_len: int,
                                          max_goals: int, rotate_objects: ObjectRotation,
                                          if_exist: ExistBehavior, num_retries: int,
                                          seed: Optional[int]=None, verbose: int=0,
                                          **kwargs: Any) -> SequentialObjectNavDatasetV0:
    out_path, dataset, idx0 = check_existence(cfg, if_exist)
    new_episodes = []
    rng = np.random.default_rng(seed)
    scene_pool = create_scene_pool(scenes_dir)
    rng.shuffle(scene_pool)
    object_pool = create_object_pool_v2(objects_dir)
    rng.shuffle(object_pool)
    view_pts_cfg = cfg.TASK.SPAWNED_OBJECTGOAL_APPEARANCE_SENSOR.VIEW_POINTS
    if view_pts_cfg.NUM_ANGLES * view_pts_cfg.NUM_RADII > 256:
        raise ValueError("Too many potential view points around a single goal (max = 256); "
                         "please update your view points config")

    num_ep_per_scene, more_ep = divmod(num_episodes, len(scene_pool))
    with tqdm.tqdm(total=num_episodes, disable=(verbose != 1)) as progress:
        for k, scene in enumerate(scene_pool):
            if num_ep_per_scene == 0 and k >= more_ep:
                break
            cfg.SIMULATOR.defrost()
            cfg.SIMULATOR.SCENE = scene
            cfg.freeze()
            scene_name = os.path.basename(scene)[:-len(DEFAULT_SCENE_PATH_EXT)]
            with habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR) as sim:
                if seed is not None:
                    sim.seed(seed + k)
                idx = 0
                sim.get_object_template_manager().load_configs(
                    os.path.join(objects_dir, ".configs")
                )
                for _ in range(num_ep_per_scene + (1 if k < more_ep else 0)):
                    try:
                        episode = generate_sequential_objectnav_episode(
                                sim, f"{scene_name}_{idx0 + idx}", min_seq_len, max_seq_len,
                                max_goals, object_pool, rotate_objects,
                                view_pts_cfg, num_retries, rng
                        )
                        new_episodes.append(episode)
                        idx += 1
                        progress.update()
                    except MaxRetriesError as e:
                        _logger.error(e)
    dataset.episodes.extend(new_episodes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, 'wt') as f:
        f.write(dataset.to_json())
    _logger.info(f"Wrote {len(new_episodes)} episodes to '{out_path}'.")
    return dataset


_DEFAULT_ARGS: Dict[str, Any] = {"config_path": "configs/tasks/pointnav_gibson.yaml",
                                 "scenes_dir": "data/scene_datasets/gibson",
                                 "objects_dir": "data/object_datasets/test_objects",
                                 "num_episodes": 25,
                                 "min_seq_len": 3,
                                 "max_seq_len": 3,
                                 "max_goals": 1,
                                 "rotate_objects": ObjectRotation.VERTICAL,
                                 "if_exist": ExistBehavior.ABORT,
                                 "num_retries": 4,
                                 "seed": None,
                                 "verbose": 0}


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
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("extra_config", nargs=argparse.REMAINDER)
    parser.set_defaults(**_DEFAULT_ARGS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose == 0:
        habitat.logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        habitat.logger.setLevel(logging.WARN)
    elif args.verbose == 2:
        habitat.logger.setLevel(logging.INFO)
    elif args.verbose == 3:
        habitat.logger.setLevel(logging.DEBUG)
    cfg = habitat.get_config(args.config_path, args.extra_config)
    generate_sequential_objectnav_dataset(cfg, **vars(args))
