from typing import Any, Optional, Tuple, List, Dict, Iterator
import argparse
import enum
import sys
import os
import itertools
import gzip

import numpy as np
from scipy.ndimage import gaussian_filter
import quaternion
import magnum as mn

import habitat
from habitat.core.simulator import Simulator
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, ViewPoint, \
                                                SpawnedObjectNavEpisode
from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset \
        import SpawnedObjectNavDatasetV0
from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution
from habitat.datasets.spawned_objectnav.utils import DEFAULT_SCENE_PATH_PREFIX, \
                                                     DEFAULT_SCENE_PATH_EXT, \
                                                     DEFAULT_OBJECT_PATH_PREFIX, \
                                                     DEFAULT_OBJECT_PATH_EXT
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


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


class ExistBehavior(enum.Enum):
    ABORT = enum.auto()
    OVERRIDE = enum.auto()
    APPEND = enum.auto()


class UnreachableGoalError(Exception):
    goal: SpawnedObjectGoal
    start_pos: List[float]

    def __init__(self, goal: SpawnedObjectGoal, start_pos: List[float]) -> None:
        super().__init__(goal, start_pos)
        self.goal = goal
        self.start_pos = start_pos

    def __str__(self) -> str:
        cat = self.goal.object_template_id.split('/')[-2]
        return f"Could not find view points reachable from {self.start_pos} " \
                + f"for goal '{cat}' at position {self.goal.position}."


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
    return sorted(ObjectPoolCategory(
        dir_entry.name, i,
        sorted(entry.path for entry in os.scandir(dir_entry.path)
               if entry.is_file() and entry.name.endswith(DEFAULT_OBJECT_PATH_EXT))
    ) for i, dir_entry in enumerate(os.scandir(objects_dir)) if dir_entry.is_dir())


def create_scene_pool(scenes_dir: str) -> List[str]:
    return sorted(entry.path for entry in os.scandir(scenes_dir)
                  if entry.is_file() and entry.name.endswith(DEFAULT_SCENE_PATH_EXT))


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
        mngr_id, = mngr.load_configs(tmpl_id)
        obj_id = sim.add_object(mngr_id)
        sim.set_translation(obj_pos, obj_id)
        sim.set_rotation(obj_rot, obj_id)
        sim.set_object_motion_type(MotionType.STATIC, obj_id)

        goal = SpawnedObjectGoal(position=obj_pos.tolist(),
                                 rotation=[*obj_rot.vector, obj_rot.scalar],
                                 object_template_id=tmpl_id,
                                 view_points=[])
        goal._spawned_object_id = obj_id
        goals.append(goal)
    return goals


def recompute_navmesh_with_static_objects(sim):
    ag_cfg = getattr(sim.habitat_config, f"AGENT_{sim.habitat_config.DEFAULT_AGENT_ID}",
                     sim.habitat_config.AGENT_0)
    settings = NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = ag_cfg.RADIUS
    settings.agent_height = ag_cfg.HEIGHT
    sim.recompute_navmesh(sim.pathfinder, settings, True)


def find_view_points(sim: Simulator, goals: List[SpawnedObjectGoal], start_pos: np.ndarray,
                     min_radius: float=0.5, max_radius: float=3.0,
                     num_radii: int=5, num_angles: int=12,
                     roi: Optional[Tuple[slice, slice]]=None,
                     iou_thresh: Optional[float]=None) -> List[SpawnedObjectGoal]:
    sensor_cfg = sim.habitat_config.DEPTH_SENSOR
    if roi is None:
        h = sensor_cfg.HEIGHT
        w = sensor_cfg.WIDTH
        s = h // 2
        roi = (slice((h - s) // 2, (h + s) // 2 + 1), slice((w - s) // 2, (w + s) // 2 + 1))

    angles = np.linspace(0, 2 * np.pi, num_angles)
    radii = np.linspace(min_radius, max_radius, num_radii)
    rel_positions = np.zeros((num_angles * num_radii, 3))
    rel_positions[:, 0] = np.outer(radii, np.cos(angles)).flatten()
    rel_positions[:, 2] = np.outer(radii, np.sin(angles)).flatten()

    sensor_pos = np.array(sensor_cfg.POSITION)
    rel_sensor_positions = rel_positions + sensor_pos

    pan = np.arctan2(rel_sensor_positions[:, 0], rel_sensor_positions[:, 2])
    pan_q = np.zeros((num_angles * num_radii, 4))
    pan_q[:, 0] = np.cos(0.5 * pan)
    pan_q[:, 2] = np.sin(0.5 * pan)
    pan_q = quaternion.from_float_array(pan_q)

    hypot = np.hypot(rel_sensor_positions[:, 0], rel_sensor_positions[:, 2])
    tilt = np.arctan(-rel_sensor_positions[:, 1] / hypot)
    tilt_q = np.zeros((num_angles * num_radii, 4))
    tilt_q[:, 0] = np.cos(0.5 * tilt)
    tilt_q[:, 1] = np.sin(0.5 * tilt)
    tilt_q = quaternion.from_float_array(tilt_q)

    sensor_rotations = pan_q * tilt_q

    s = sim.get_agent_state()
    max_y = sim.pathfinder.get_bounds()[1][1]

    for goal in goals:
        obj_pos = np.array(goal.position)
        for rel_pos, rot in zip(rel_positions, sensor_rotations):
            pos = obj_pos + rel_pos
            if not sim.pathfinder.is_navigable(pos):
                continue
            if not np.isfinite(sim.geodesic_distance(start_pos, pos)):
                continue

            s.sensor_states['depth'].position = pos + sensor_pos
            s.sensor_states['depth'].rotation = rot
            sim.get_agent(0).set_state(s, False, False)
            depth_with = sim.get_sensor_observations()['depth']

            sim.set_object_motion_type(MotionType.KINEMATIC, goal._spawned_object_id)
            sim.set_translation([0.0, 2 * max_y, 0.0], goal._spawned_object_id)
            depth_without = sim.get_sensor_observations()['depth']
            sim.set_translation(goal.position, goal._spawned_object_id)
            sim.set_object_motion_type(MotionType.STATIC, goal._spawned_object_id)

            diff = (depth_with != depth_without)
            diff_roi = diff[roi]
            iou = diff_roi.sum() / diff_roi.size
            if (iou_thresh is not None and iou < iou_thresh) or not diff.any():
                continue
            goal.view_points.append(ViewPoint(position=(pos + sensor_pos).tolist(),
                                              rotation=[rot.x, rot.y, rot.z, rot.w],
                                              iou=iou))
        if not goal.view_points:
            raise UnreachableGoalError(goal, start_pos)
    return goals


def generate_spawned_objectgoals(sim: Simulator, start_pos: np.ndarray,
                                 template_ids: List[str], rotate_objects: ObjectRotation,
                                 rng: np.random.Generator) -> List[SpawnedObjectGoal]:
    distrib = SpawnPositionDistribution(sim, height=start_pos[1])
    positions = distrib.sample_reachable_from_position(len(template_ids), start_pos, rng)
    goals = spawn_objects(sim, template_ids, positions, rotate_objects, rng)
    recompute_navmesh_with_static_objects(sim)
    goals = find_view_points(sim, goals, start_pos)
    return goals


def generate_spawned_objectnav_episode(sim: Simulator,
                                       ep_id: str,
                                       max_goals: int,
                                       object_pool: List[ObjectPoolCategory],
                                       rotate_objects: ObjectRotation,
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
            goals = generate_spawned_objectgoals(sim, start_pos, tmpl_ids, rotate_objects, rng)
            break
        except UnreachableGoalError as e:
            errors.append(e)
    else:
        raise MaxRetriesError("generate reachable goals", num_retries, errors)

    return SpawnedObjectNavEpisode(episode_id=ep_id,
                                   scene_id=sim.habitat_config.SCENE,
                                   start_position=start_pos,
                                   start_rotation=start_rot,
                                   object_category=category,
                                   object_category_index=cat_index,
                                   goals=goals)


def generate_spawned_objectnav_dataset(config_path: str, extra_config: List[str],
                                       scenes_dir: str, objects_dir: str,
                                       num_episodes:int, max_goals: int,
                                       rotate_objects: ObjectRotation,
                                       if_exist: ExistBehavior,
                                       num_retries: int,
                                       seed: Optional[int]=None) -> SpawnedObjectNavDatasetV0:
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
    ep_id_gen = (f"episode_{i}" for i in itertools.count())

    rng = np.random.default_rng(seed)
    scene_pool = create_scene_pool(scenes_dir)
    rng.shuffle(scene_pool)
    object_pool = create_object_pool(objects_dir)
    rng.shuffle(object_pool)

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
                episode = generate_spawned_objectnav_episode(sim, next(ep_id_gen), max_goals,
                                                             object_pool, rotate_objects,
                                                             num_retries, rng)
                new_episodes.append(episode)
    dataset.episodes.extend(new_episodes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, 'wt') as f:
        f.write(dataset.to_json())
    return dataset


_DEFAULT_ARGS: Dict[str, Any] = {"config_path": "configs/tasks/spawned_objectnav.yaml",
                                 "scenes_dir": "data/scene_datasets/habitat-test-scenes",
                                 "objects_dir": "data/object_datasets/test_objects",
                                 "num_episodes": 25,
                                 "max_goals": 2,
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
    generate_spawned_objectnav_dataset(**vars(_parse_args()))
