from typing import Any, Optional, Tuple, List, Dict, Iterator
import argparse
import os
import glob
import gzip
import time
import logging
import enum
import itertools
import random

import numpy as np
import quaternion
import magnum as mn
import cv2
import tqdm

import habitat
from habitat.config.default import Config
from habitat.core.simulator import Simulator
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode
from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset \
        import SpawnedObjectNavDatasetV0
from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution
from habitat.datasets.spawned_objectnav.utils import (DEFAULT_SCENE_PATH_EXT,
                                                      DEFAULT_OBJECT_PATH_EXT,
                                                      get_uniform_view_pt_positions,
                                                      get_view_pt_rotations,
                                                      render_view_pts,
                                                      DebugMapRenderer)
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


_filename, _ = os.path.splitext(os.path.basename(__file__))
_logger = habitat.logger.getChild(_filename)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[{asctime}] {levelname} ({name}): {msg}", style='{'))
_logger.addHandler(_handler)
_debug_map_renderer = DebugMapRenderer()


class ObjectPoolCategory:
    name: str
    index: int
    templates: List[str]

    def __init__(self, name, index, templates) -> None:
        self.name = name
        self.index = index
        self.templates = sorted(templates)

    def __lt__(self, other) -> bool:
        return self.name < other.name

    def __iter__(self) -> Iterator[Any]:
        yield self.name
        yield self.index
        yield self.templates


class ObjectRotation(enum.Enum):
    FIXED = enum.auto()
    VERTICAL = enum.auto()
    FREE = enum.auto()

    def __str__(self):
        return self.name


class ExistBehavior(enum.Enum):
    ABORT = enum.auto()
    OVERRIDE = enum.auto()
    APPEND = enum.auto()

    def __str__(self):
        return self.name


class UnreachableGoalError(Exception):
    goal: SpawnedObjectGoal
    start_pos: List[float]

    def __init__(self, goal: SpawnedObjectGoal, start_pos: List[float]) -> None:
        super().__init__(goal, start_pos)
        self.goal = goal
        self.start_pos = start_pos

    def __str__(self) -> str:
        cat, mdl = self.goal.object_template_id.split('/')[-2:]
        tmpl = '/'.join((cat, mdl[:-len(DEFAULT_OBJECT_PATH_EXT)]))
        s_pos_str = '[' + ','.join(f"{x:.3f}" for x in self.start_pos) + ']'
        g_pos_str = '[' + ','.join(f"{x:.3f}" for x in self.goal.position) + ']'
        return f"Could not find view points reachable from {s_pos_str} " \
                + f"for goal '{tmpl}' at position {g_pos_str}."


class MaxRetriesError(Exception):
    task_desc: str
    num_retries: int
    errors: List[Exception]

    def __init__(self, task_desc: str, num_retries: int, errors: List[Exception]) -> None:
        super().__init__(task_desc, num_retries, errors)
        self.task_desc = task_desc
        self.num_retries = num_retries
        self.errors = errors

    def __str__(self) -> str:
        return f"Could not {self.task_desc} after {self.num_retries} retries:\n" \
                + "\n".join(f"  {i}. {e!s}" for i, e in enumerate(self.errors, start=1))


def create_object_pool(objects_dir: str) -> List[ObjectPoolCategory]:
    pool = sorted(ObjectPoolCategory(
        dir_entry.name, i,
        sorted(entry.path for entry in os.scandir(dir_entry.path)
               if entry.is_file() and entry.name.endswith(DEFAULT_OBJECT_PATH_EXT))
    ) for i, dir_entry in enumerate(os.scandir(objects_dir)) if dir_entry.is_dir())
    _logger.debug(f"Created pool of {len(pool)} object categories.")
    return pool


def create_scene_pool(scenes_dir: str) -> List[str]:
    pool = sorted(glob.glob(os.path.join(scenes_dir, "**", f"*{DEFAULT_SCENE_PATH_EXT}"),
                            recursive=True))
    _logger.debug(f"Created pool of {len(pool)} scenes.")
    return pool


def spawn_objects(sim: Simulator,
                  template_ids: List[str],
                  positions: np.ndarray,
                  rotate_objects: ObjectRotation=ObjectRotation.FIXED,
                  rng: Optional[np.random.Generator]=None) -> List[SpawnedObjectGoal]:
    num_objects = len(template_ids)
    mngr = sim.get_object_template_manager()

    if rotate_objects is ObjectRotation.FIXED:
        rotations = [mn.Quaternion.identity_init() for _ in range(num_objects)]
    elif rotate_objects is ObjectRotation.VERTICAL:
        if rng is None:
            rng = np.random.default_rng()
        angles = 2 * np.pi * rng.random(num_objects)
        rotations = [mn.Quaternion.rotation(mn.Rad(a), mn.Vector3(*sim.up_vector))
                     for a in angles]
    elif rotate_objects is ObjectRotation.FREE:
        if rng is None:
            rng = np.random.default_rng()
        angles = 2 * np.pi * rng.random(num_objects)
        axes = rng.normal(size=(num_objects, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        rotations = [mn.Quaternion.rotation(mn.Rad(a), mn.Vector3(*axis))
                     for a, axis in zip(angles, axes)]

    goals = []
    for obj_pos, obj_rot, tmpl_id in zip(positions, rotations, template_ids):
        goal = SpawnedObjectGoal(position=obj_pos.tolist(),
                                 rotation=[*obj_rot.vector, obj_rot.scalar],
                                 object_template_id=tmpl_id,
                                 valid_view_points_indices=[])
        goal._spawn_in_sim(sim)
        goals.append(goal)

    _logger.debug(f"Spawned {num_objects} objects "
                  + "with {rotate_objects.name.lower()} rotations in simulator.")
    return goals


def recompute_navmesh_with_static_objects(sim):
    ag_cfg = getattr(sim.habitat_config, f"AGENT_{sim.habitat_config.DEFAULT_AGENT_ID}",
                     sim.habitat_config.AGENT_0)
    settings = NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = ag_cfg.RADIUS
    settings.agent_height = ag_cfg.HEIGHT
    sim.recompute_navmesh(sim.pathfinder, settings, True)
    _logger.debug(f"Recomputed simulator navmesh for agent with radius {ag_cfg.RADIUS:.2f}m "
                  + "and height {ag_cfg.HEIGHT:.2f}m.")


def check_reachability(sim: Simulator, goals: List[SpawnedObjectGoal],
                       start_pos: np.ndarray) -> None:
    for goal in goals:
        if not np.isfinite(sim.geodesic_distance(start_pos, goal.pathfinder_targets)):
            raise UnreachableGoalError(goal, start_pos)
    else:
        _logger.debug(f"All {len(goals)} goals are reachable.")


def find_view_points(sim: Simulator, goals: List[SpawnedObjectGoal], start_pos: np.ndarray,
                     view_pts_cfg: Config, roi: Optional[Tuple[slice, slice]]=None,
                     iou_thresh: Optional[float]=None) -> List[SpawnedObjectGoal]:
    sensor_cfg = sim.habitat_config.DEPTH_SENSOR
    if roi is None:
        h = sensor_cfg.HEIGHT
        w = sensor_cfg.WIDTH
        s = h // 2
        roi_vert = slice((h - s) // 2, (h + s) // 2 + 1)
        roi_horz = slice((w - s) // 2, (w + s) // 2 + 1)
    else:
        roi_vert, roi_horz = roi

    rel_positions = get_uniform_view_pt_positions(**{key.lower(): val
                                                     for key, val in view_pts_cfg.items()})
    sensor_pos = np.array(sensor_cfg.POSITION)
    rel_sensor_positions = rel_positions + sensor_pos
    abs_sensor_rotations = get_view_pt_rotations(rel_sensor_positions)
    _, (_, max_y, _) = sim.pathfinder.get_bounds()

    for goal in goals:
        obj_pos = np.array(goal.position)
        positions = obj_pos + rel_positions
        reachable = np.array([sim.pathfinder.is_navigable(pos) \
                              and np.isfinite(sim.geodesic_distance(start_pos, pos))
                              for pos in positions])
        if not reachable.any():
            raise UnreachableGoalError(goal, start_pos)

        cand_positions = obj_pos + rel_sensor_positions[reachable]
        cand_rotations = abs_sensor_rotations[reachable]
        depth_with = render_view_pts(sim, cand_positions, cand_rotations)['depth']

        sim.set_object_motion_type(MotionType.KINEMATIC, goal._spawned_object_id)
        sim.set_translation([0.0, 2 * max_y, 0.0], goal._spawned_object_id)
        depth_without = render_view_pts(sim, cand_positions, cand_rotations)['depth']
        sim.set_translation(goal.position, goal._spawned_object_id)
        sim.set_object_motion_type(MotionType.STATIC, goal._spawned_object_id)

        diff = (depth_with[..., 0] != depth_without[..., 0])

        diff_roi = diff[:, roi_vert, roi_horz]
        scores = diff_roi.sum(axis=(1, 2)) / diff_roi[0].size
        visible = diff.any(axis=(1, 2)) if iou_thresh is None else scores >= iou_thresh

        cand_indices, = np.nonzero(reachable)
        goal.valid_view_points_indices = cand_indices[visible].astype(np.uint8)
        if not goal.valid_view_points_indices:
            raise UnreachableGoalError(goal, start_pos)
        goal.valid_view_points_ious = scores[visible].astype(np.float32)
    _logger.debug(f"All {len(goals)} goals have at least one view point.")
    return goals


def clear_sim_from_objects(sim):
    num_objects = 0
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)
        num_objects += 1
    sim.get_object_template_manager().remove_all_templates()
    _logger.debug(f"Cleared simulator from {num_objects} objects.")


def generate_spawned_objectgoals(sim: Simulator, start_pos: np.ndarray,
                                 template_ids: List[str], rotate_objects: ObjectRotation,
                                 view_pts_cfg: Config,
                                 rng: np.random.Generator) -> List[SpawnedObjectGoal]:
    clear_sim_from_objects(sim)
    distrib = SpawnPositionDistribution(sim, height=start_pos[1])
    positions = distrib.sample_reachable_from_position(len(template_ids), start_pos, rng)
    goals = spawn_objects(sim, template_ids, positions, rotate_objects, rng)
    recompute_navmesh_with_static_objects(sim)
    try:
        check_reachability(sim, goals, start_pos)
        goals = find_view_points(sim, goals, start_pos, view_pts_cfg)
    except UnreachableGoalError as e:
        if _logger.isEnabledFor(logging.DEBUG):
            _debug_map_renderer.render(sim, distrib, goals, e)
        raise
    return goals


def generate_spawned_objectnav_episode(sim: Simulator,
                                       ep_id: str,
                                       max_goals: int,
                                       object_pool: List[ObjectPoolCategory],
                                       rotate_objects: ObjectRotation,
                                       view_pts_cfg: Config,
                                       num_retries: int,
                                       rng: np.random.Generator) -> SpawnedObjectNavEpisode:
    start_pos = sim.sample_navigable_point()
    a = 2 * np.pi * rng.random()
    start_rot = [*(np.sin(0.5 * a) * sim.up_vector), np.cos(0.5 * a)]

    category, cat_index, tmpl_ids = rng.choice(object_pool)
    if len(tmpl_ids) > max_goals:
        tmpl_ids = rng.choice(tmpl_ids, max_goals, replace=True)
    errors = []
    for _ in range(num_retries):
        try:
            goals = generate_spawned_objectgoals(sim, start_pos, tmpl_ids, rotate_objects,
                                                 view_pts_cfg, rng)
            break
        except UnreachableGoalError as e:
            errors.append(e)
    else:
        raise MaxRetriesError("generate reachable goals", num_retries, errors)

    episode= SpawnedObjectNavEpisode(episode_id=ep_id,
                                     scene_id=sim.habitat_config.SCENE,
                                     start_position=start_pos,
                                     start_rotation=start_rot,
                                     object_category=category,
                                     object_category_index=cat_index,
                                     goals=goals)
    _logger.info(f"Successfully generated episode '{ep_id}'.")
    return episode


def check_existence(cfg: Config, if_exist: ExistBehavior):
    out_path = cfg.DATASET.DATA_PATH.format(split=cfg.DATASET.SPLIT)
    idx0 = 0
    try:
        dataset = habitat.make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
        if if_exist is ExistBehavior.ABORT:
            _logger.error(f"'{out_path}' already exists, aborting.")
            return None
        elif if_exist is ExistBehavior.OVERRIDE:
            _logger.warning(f"Overriding '{out_path}'.")
            dataset.episodes = []
        elif if_exist is ExistBehavior.APPEND:
            idx0 = len(dataset.episodes)
            _logger.info(f"Appending episodes to '{out_path}'.")
    except FileNotFoundError:
        _logger.info(f"Creating new dataset '{out_path}'.")
        dataset = habitat.make_dataset(cfg.DATASET.TYPE)
    return out_path, dataset, idx0


def generate_spawned_objectnav_dataset(cfg: Config, scenes_dir: str, objects_dir: str,
                                       num_episodes:int, max_goals: int,
                                       rotate_objects: ObjectRotation,
                                       if_exist: ExistBehavior, num_retries: int,
                                       seed: Optional[int]=None, verbose: int=0,
                                       **kwargs) -> SpawnedObjectNavDatasetV0:
    out_path, dataset, idx0 = check_existence(cfg, if_exist)
    new_episodes = []
    rng = np.random.default_rng(seed)
    scene_pool = create_scene_pool(scenes_dir)
    rng.shuffle(scene_pool)
    object_pool = create_object_pool(objects_dir)
    rng.shuffle(object_pool)
    view_pts_cfg = cfg.TASK.SPAWNED_OBJECTGOAL_APPEARANCE.VIEW_POINTS
    if view_pts_cfg.NUM_ANGLES * view_pts_cfg.NUM_RADII > 256:
        raise ValueError("Too many potential view points around a single goal (max = 256); "
                         "please update your view points config")

    num_ep_per_scene, more_ep = divmod(num_episodes, len(scene_pool))
    with tqdm.tqdm(total=num_episodes, disable=(verbose!=1)) as progress:
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
                for _ in range(num_ep_per_scene + (1 if k < more_ep else 0)):
                    try:
                        episode = generate_spawned_objectnav_episode(
                                sim, f"{scene_name}_{idx0 + idx}", max_goals,
                                object_pool, rotate_objects, view_pts_cfg, num_retries, rng
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


_DEFAULT_ARGS: Dict[str, Any] = {"config_path": "configs/tasks/spawned_objectnav.yaml",
                                 "scenes_dir": "data/scene_datasets/habitat-test-scenes",
                                 "objects_dir": "data/object_datasets/test_objects",
                                 "num_episodes": 25,
                                 "max_goals": 2,
                                 "rotate_objects": ObjectRotation.FIXED,
                                 "if_exist": ExistBehavior.ABORT,
                                 "num_retries": 4,
                                 "seed": None,
                                 "verbose": 0}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", "-c")
    parser.add_argument("--scenes-dir")
    parser.add_argument("--objects-dir")
    parser.add_argument("--num-episodes", "-n", type=int)
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
    return parser.parse_args()


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
    generate_spawned_objectnav_dataset(cfg, **vars(args))
