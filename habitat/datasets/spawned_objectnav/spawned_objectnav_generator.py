from typing import Any, Optional, Tuple, List, Set, Dict
import argparse
import sys
import os
import collections
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
from habitat.datasets.spawned_objectnav.utils import DEFAULT_SCENE_PATH_PREFIX, \
                                                     DEFAULT_SCENE_PATH_EXT, \
                                                     DEFAULT_OBJECT_PATH_PREFIX, \
                                                     DEFAULT_OBJECT_PATH_EXT
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


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


class SpawnPositionDistribution:
    _sim: Simulator
    _height: float
    _resolution: float
    _margin: float
    _inflate_radius: float
    _nav_mask: Optional[np.ndarray]
    _origin: Optional[np.ndarray]
    _edges: Optional[np.ndarray]
    _distrib: Optional[np.ndarray]
    _cumul: Optional[np.ndarray]
    _rng: Optional[np.random.Generator]

    def __init__(self, sim: Simulator, height: Optional[float]=None,
                       resolution: float=0.02, margin: float=0.1,
                       inflate_radius: float=1.8, seed: Optional[int]=None) -> None:
        self._sim = sim
        self._height = sim.get_agent_state().position[1] if height is None else height
        self._resolution = resolution
        self._margin = margin
        self._inflate_radius = inflate_radius

        self._nav_mask = None
        self._origin = None
        self._edges = None
        self._distrib = None
        self._cumul = None
        self._rng = None

        self._update_navmesh()
        if seed is not None:
            self.seed(seed)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def set_height(self, height: float) -> None:
        self._height = height
        self._update_edges()

    @property
    def resolution(self) -> float:
        return self._resolution

    @resolution.setter
    def set_resolution(self, resolution: float) -> None:
        self._resolution = resolution
        self._update_edges()

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def set_margin(self, margin: float) -> None:
        self._margin = margin
        self._update_navmesh()

    @property
    def inflate_radius(self) -> float:
        return self._inflate_radius

    @inflate_radius.setter
    def set_inflate_radius(self, inflate_radius: float) -> None:
        self._inflate_radius = inflate_radius
        self._update_distrib()

    def _update_navmesh(self) -> None:
        settings = NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = self._margin
        settings.agent_height = self._margin
        self._sim.recompute_navmesh(self._sim.pathfinder, settings)
        self._origin, _ = self._sim.pathfinder.get_bounds()
        self._origin[1] = self._height
        self._update_edges()

    def _update_edges(self) -> None:
        self._nav_mask = self._sim.pathfinder.get_topdown_view(self._resolution, self._height)
        edges = (self._nav_mask[:-1, :-1] != self._nav_mask[:-1, 1:]) \
                | (self._nav_mask[:-1, :-1] != self._nav_mask[1:, :-1])
        self._edges = edges.astype(np.float32)
        self._update_distrib()

    def _update_distrib(self) -> None:
        ker_sigma = 0.2 * self._inflate_radius / self._resolution
        distrib = gaussian_filter(self._edges, ker_sigma, mode='constant', truncate=5)
        distrib[~self._nav_mask[:-1, :-1]] = 0
        self._distrib = distrib
        self._cumul = self._distrib.flatten().cumsum()

    def get_origin(self) -> np.ndarray:
        return self._origin

    def get_map_edges(self) -> np.ndarray:
        return self._edges

    def get_spatial_distribution(self) -> np.ndarray:
        return self._distrib

    def seed(self, seed: Optional[int]=None) -> None:
        self._rng = np.random.default_rng(seed)

    def world_to_map(self, world_xyz: np.ndarray) -> np.ndarray:
        return ((world_xyz - self._origin) / self._resolution).astype(np.int64)[:, [2, 0]]

    def map_to_world(self, map_ij: np.ndarray,
                           map_j: Optional[np.ndarray]=None) -> np.ndarray:
        if map_j is None:
            map_i = map_ij[:, 0]
            map_j = map_ij[:, 1]
        else:
            map_i = map_ij
        world_xyz = np.tile(self._origin, (map_i.shape[0], 1))
        world_xyz[:, 0] += self._resolution * map_j
        world_xyz[:, 2] += self._resolution * map_i
        return world_xyz

    def sample(self, num_samples: int=1,
                     rng: Optional[np.random.Generator]=None) -> np.ndarray:
        if rng is None:
            if self._rng is None:
                self.seed()
            rng = self._rng
        u = self._cumul[-1] * rng.random(num_samples)
        flat_i = np.digitize(u, self._cumul)
        i, j = np.unravel_index(flat_i, self._distrib.shape)
        return self.map_to_world(i, j)


def spawn_objects(sim: Simulator, template_ids: List[str],
                  positions: np.ndarray, rotate_objects: str="DISABLE",
                  rng: Optional[np.random.Generator]=None) -> None:
    num_objects = len(template_ids)
    mngr = sim.get_object_template_manager()

    if rotate_objects == "DISABLE":
        rotations = [mn.Quaternion.identity_init() for _ in range(num_objects)]
    elif rotate_objects == "VERTICAL":
        if rng is None:
            rng = np.random.default_rng()
        angles = 2 * np.pi * rng.random(num_objects)
        rotations = [mn.Quaternion.rotation(mn.Rad(a), mn.Vector3(*sim.up_vector))
                     for a in angles]
    elif rotate_objects == "3D":
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
                     iou_thresh: Optional[float]=None) -> None:
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
    hypot = np.hypot(rel_sensor_positions[:, 0], rel_sensor_positions[:, 2])
    tilt = np.arctan(-rel_sensor_positions[:, 1] / hypot)
    sensor_rotations = quaternion.from_euler_angles(tilt, pan, np.zeros_like(pan))

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
    return goals


def generate_spawned_objectgoals(sim: Simulator, template_ids: List[str],
                                 rotate_objects: str, start_pos: np.ndarray,
                                 rng: np.random.Generator) -> List[SpawnedObjectGoal]:
    distrib = SpawnPositionDistribution(sim)
    positions = distrib.sample(len(template_ids), rng)
    goals = spawn_objects(sim, template_ids, positions, rotate_objects, rng)
    recompute_navmesh_with_static_objects(sim)
    goals = find_view_points(sim, goals, start_pos)
    return goals


def generate_spawned_objectnav_episode(sim: Simulator,
                                       object_pool: Dict[str, Set[str]],
                                       category_index_map: Dict[str, int],
                                       ep_id: str, rng: np.random.Generator, max_goals: int,
                                       rotate_objects: str) -> SpawnedObjectNavEpisode:
    start_pos = sim.sample_navigable_point()
    a = 2 * np.pi * rng.random()
    start_rot = [*(np.sin(0.5 * a) * sim.up_vector), np.cos(0.5 * a)]

    category, tmpl_ids = rng.choice(list(object_pool.items()), replace=False)
    cat_index = category_index_map[category]
    if len(tmpl_ids) > max_goals:
        tmpl_ids = rng.choice(list(tmpl_ids), max_goals, replace=True)
    goals = generate_spawned_objectgoals(sim, tmpl_ids, rotate_objects, start_pos, rng)

    return SpawnedObjectNavEpisode(episode_id=ep_id,
                                   scene_id=sim.habitat_config.SCENE,
                                   start_position=start_pos,
                                   start_rotation=start_rot,
                                   object_category=category,
                                   object_category_index=cat_index,
                                   goals=goals)


def generate_spawned_objectnav_dataset(config_path: str, extra_config: List[str],
                                       num_episodes:int, max_goals: int, rotate_objects: str,
                                       if_exist: str, scenes_dir: str, objects_dir: str,
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
                                                             max_goals, rotate_objects)
                new_episodes.append(episode)
    dataset.episodes.extend(new_episodes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, 'wt') as f:
        f.write(dataset.to_json())


_DEFAULT_ARGS: Dict[str, Any] = {"config_path": "configs/tasks/pointnav_gibson.yaml",
                                 "seed": None,
                                 "num_episodes": 4000,
                                 "max_goals": 5,
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
    parser.add_argument("--rotate-objects", choices=("DISABLE", "YAXIS", "3D"))
    parser.add_argument("--if-exist", choices=("ABORT", "OVERRIDE", "APPEND"))
    parser.add_argument("--scenes-dir")
    parser.add_argument("--objects-dir")
    parser.add_argument("extra_config", nargs=argparse.REMAINDER)
    parser.set_defaults(**_DEFAULT_ARGS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate_spawned_objectnav_dataset(**vars(_parse_args()))
