from typing import Any, Optional, Tuple, List, Dict, Iterator
import argparse
import enum
import sys
import os
import glob
import itertools
import gzip

import numpy as np
from scipy.ndimage import gaussian_filter
import quaternion
import magnum as mn
import tqdm

import habitat
from habitat.core.simulator import Simulator
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, ViewPoint, \
                                                SpawnedObjectNavEpisode
from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset \
        import SpawnedObjectNavDatasetV0
from habitat.datasets.spawned_objectnav.spawn_pos_distrib import SpawnPositionDistribution
from habitat.datasets.spawned_objectnav.utils import DEFAULT_SCENE_PATH_EXT, \
                                                     DEFAULT_OBJECT_PATH_EXT, \
                                                     get_uniform_view_pt_positions, \
                                                     get_view_pt_rotations, \
                                                     render_view_pts
from habitat_sim.nav import NavMeshSettings
from habitat_sim.physics import MotionType


_debug_cnt = 0
def _debug_render(sim, distrib, goals, error):
    import cv2

    def put_text(disp, txt, emph=False):
        thick = 2 if emph else 1
        (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, thick)
        cv2.putText(disp, txt, (j - w // 2, i - b),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thick)

    d = distrib.get_spatial_distribution()
    m = distrib.get_nav_mask()
    disp = cv2.applyColorMap((d * 255 / d.max()).astype(np.uint8), cv2.COLORMAP_JET)
    disp[~m] = 0
    new_m = sim.pathfinder.get_topdown_view(distrib.resolution, distrib.height)
    disp[m & ~new_m] //= 2
    i, j = distrib.world_to_map(error.start_pos)
    cv2.circle(disp, (j, i), 5, (0, 255, 0), 2)
    put_text(disp, "START", False)
    for goal in goals:
        cat = goal.object_template_id.split('/')[-2]
        i, j = distrib.world_to_map(np.array(goal.position))
        cv2.circle(disp, (j, i), 5, (0, 0, 255), 2)
        put_text(disp, cat, goal is error.goal)

        t, l = distrib.world_to_map(goal.position + np.array(goal._bounding_box.min))
        b, r = distrib.world_to_map(goal.position + np.array(goal._bounding_box.max))
        cv2.rectangle(disp, (l, t), (r, b), (225, 0, 255))

        for view_pt in goal.view_points:
            i, j = distrib.world_to_map(view_pt.position)
            cv2.circle(disp, (j, i), 3, (0, 255, 255), 2)

    global _debug_cnt
    outpath = f"TEMP/DEBUG_render_{_debug_cnt}.png"
    _debug_cnt += 1
    cv2.imwrite(outpath, disp)


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
        cat = self.goal.object_template_id.split('/')[-2]
        s_pos_str = '[' + ','.join(f"{x:.3f}" for x in self.start_pos) + ']'
        g_pos_str = '[' + ','.join(f"{x:.3f}" for x in self.goal.position) + ']'
        return f"Could not find view points reachable from {s_pos_str} " \
                + f"for goal '{cat}' at position {g_pos_str}."


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
    return sorted(glob.glob(os.path.join(scenes_dir, "**", f"*{DEFAULT_SCENE_PATH_EXT}"),
                            recursive=True))


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
        obj_node = sim.get_object_scene_node(obj_id)
        shift = np.array([0, obj_node.cumulative_bb.bottom, 0])
        sim.set_translation(obj_pos - shift, obj_id)
        sim.set_rotation(obj_rot, obj_id)
        sim.set_object_motion_type(MotionType.STATIC, obj_id)

        goal = SpawnedObjectGoal(position=obj_pos.tolist(),
                                 rotation=[*obj_rot.vector, obj_rot.scalar],
                                 object_template_id=tmpl_id,
                                 view_points=[])
        goal._spawned_object_id = obj_id
        goal._bounding_box = obj_node.cumulative_bb
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


def check_reachability(sim: Simulator, goals: List[SpawnedObjectGoal],
                       start_pos: np.ndarray) -> None:
    for goal in goals:
        bb = goal._bounding_box
        shifts = np.array([[x, 0, z] for x in (0, bb.left, bb.right)
                                     for z in (0, bb.back, bb.front)])
        targets = np.array(goal.position)[None, :] + shifts
        if not np.isfinite(sim.geodesic_distance(start_pos, targets)):
            raise UnreachableGoalError(goal, start_pos)


def find_view_points(sim: Simulator, goals: List[SpawnedObjectGoal], start_pos: np.ndarray,
                     num_angles: int=20, num_radii: int=10,
                     min_radius: float=0.5, max_radius: float=3.5,
                     roi: Optional[Tuple[slice, slice]]=None,
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

    rel_positions = get_uniform_view_pt_positions(num_angles, num_radii, min_radius, max_radius)
    sensor_pos = np.array(sensor_cfg.POSITION)
    rel_sensor_positions = rel_positions + sensor_pos

    abs_sensor_rotations = get_view_pt_rotations(rel_sensor_positions)

    max_y = sim.pathfinder.get_bounds()[1][1]

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
        sim.set_translation([0.0, 3 * max_y, 0.0], goal._spawned_object_id)
        depth_without = render_view_pts(sim, cand_positions, cand_rotations)['depth']
        sim.set_translation(goal.position, goal._spawned_object_id)
        sim.set_object_motion_type(MotionType.STATIC, goal._spawned_object_id)

        diff = (depth_with != depth_without)

        diff_roi = diff[:, roi_vert, roi_horz]
        scores = diff_roi.sum(axis=(1, 2)) / diff_roi[0].size
        visible = diff.any(axis=(1, 2)) if iou_thresh is None else scores >= iou_thresh

        goal.view_points = [ViewPoint(position=pos.tolist(),
                                      rotation=[rot.x, rot.y, rot.z, rot.w],
                                      iou=iou)
                            for pos, rot, iou in zip(cand_positions[visible],
                                                     cand_rotations[visible],
                                                     scores[visible])]
        if not goal.view_points:
            raise UnreachableGoalError(goal, start_pos)
    return goals


def clear_sim_from_objects(sim):
    for obj_id in sim.get_existing_object_ids():
        sim.remove_object(obj_id)
    sim.get_object_template_manager().remove_all_templates()


def generate_spawned_objectgoals(sim: Simulator, start_pos: np.ndarray,
                                 template_ids: List[str], rotate_objects: ObjectRotation,
                                 rng: np.random.Generator) -> List[SpawnedObjectGoal]:
    clear_sim_from_objects(sim)
    distrib = SpawnPositionDistribution(sim, height=start_pos[1])
    positions = distrib.sample_reachable_from_position(len(template_ids), start_pos, rng)
    goals = spawn_objects(sim, template_ids, positions, rotate_objects, rng)
    recompute_navmesh_with_static_objects(sim)
    try:
        check_reachability(sim, goals, start_pos)
        goals = find_view_points(sim, goals, start_pos)
    except UnreachableGoalError as e:
        _debug_render(sim, distrib, goals, e)
        raise
    return goals


def generate_spawned_objectnav_episode(sim: Simulator,
                                       ep_id: str,
                                       max_goals: int,
                                       object_pool: List[ObjectPoolCategory],
                                       rotate_objects: ObjectRotation,
                                       num_retries: int,
                                       rng: np.random.Generator) -> SpawnedObjectNavEpisode:
    height = sim.get_agent_state().position[1]
    start_pos = sim.sample_navigable_point()
    while abs(start_pos[1] - height) > 0.05:
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
                    try:
                        episode = generate_spawned_objectnav_episode(sim,
                                                                     next(ep_id_gen),
                                                                     max_goals,
                                                                     object_pool,
                                                                     rotate_objects,
                                                                     num_retries,
                                                                     rng)
                        new_episodes.append(episode)
                    except MaxRetriesError as e:
                        print(e)
                    progress.update()
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
    import logging
    habitat.logger.setLevel(logging.ERROR)
    generate_spawned_objectnav_dataset(**vars(_parse_args()))
