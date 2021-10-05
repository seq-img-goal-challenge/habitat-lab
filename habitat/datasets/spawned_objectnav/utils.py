from typing import Optional, Tuple, Dict
import os

import numpy as np
import quaternion
import cv2

import habitat
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.pointnav.pointnav_dataset import DEFAULT_SCENE_PATH_PREFIX


DEFAULT_SCENE_PATH_EXT = ".glb"
DEFAULT_OBJECT_PATH_PREFIX = "data/object_datasets/"
DEFAULT_OBJECT_PATH_EXT = ".object_config.json"

HABITAT_INSTALL_DIR = os.path.dirname(habitat.__file__)
HABITAT_SCENE_PATH_PREFIX = os.path.normpath(os.path.join(HABITAT_INSTALL_DIR, "..",
                                                          DEFAULT_SCENE_PATH_PREFIX))
HABITAT_OBJECT_PATH_PREFIX = os.path.normpath(os.path.join(HABITAT_INSTALL_DIR, "..",
                                                           DEFAULT_OBJECT_PATH_PREFIX))


def find_scene_file(scene_id: str, scenes_dir: Optional[str]=None) -> str:
    candidates = ('.', DEFAULT_SCENE_PATH_PREFIX, HABITAT_SCENE_PATH_PREFIX)
    if scenes_dir is not None:
        candidates = (scenes_dir,) + candidates
    for prefix in candidates:
        for ext in ('', DEFAULT_SCENE_PATH_EXT):
            path = os.path.join(prefix, scene_id + ext)
            if os.path.isfile(path):
                return path, prefix, ext
    raise FileNotFoundError("Could not find scene file '{}'".format(scene_id))


def find_object_config_file(tmpl_id: str, objects_dir: Optional[str]=None) -> str:
    candidates = ('.', DEFAULT_OBJECT_PATH_PREFIX, HABITAT_OBJECT_PATH_PREFIX)
    if objects_dir is not None:
        candidates = (objects_dir,) + candidates
    for prefix in candidates:
        for ext in ('', DEFAULT_OBJECT_PATH_EXT):
            path = os.path.join(prefix, tmpl_id + ext)
            if os.path.isfile(path):
                return path, prefix, ext
    raise FileNotFoundError("Could not find object config file for '{}'".format(tmpl_id))


def strip_scene_id(scene_id: str, scenes_dir: Optional[str]=None) -> str:
    if scene_id.endswith(DEFAULT_SCENE_PATH_EXT):
        scene_id = scene_id[:-len(DEFAULT_SCENE_PATH_EXT)]
    candidates = ('.', DEFAULT_SCENE_PATH_PREFIX, HABITAT_SCENE_PATH_PREFIX)
    if scenes_dir is not None:
        candidates = (scenes_dir,) + candidates
    for prefix in candidates:
        if scene_id.startswith(prefix):
            scene_id = scene_id[len(prefix):].strip('/.')
            return scene_id
    return scene_id


def strip_object_template_id(obj_tmpl_id, objects_dir):
    if obj_tmpl_id.endswith(DEFAULT_OBJECT_PATH_EXT):
        obj_tmpl_id = obj_tmpl_id[:-len(DEFAULT_OBJECT_PATH_EXT)]
    candidates = ('.', DEFAULT_OBJECT_PATH_PREFIX, HABITAT_OBJECT_PATH_PREFIX)
    if objects_dir is not None:
        candidates = (objects_dir,) + candidates
    for prefix in candidates:
        if obj_tmpl_id.startswith(prefix):
            obj_tmpl_id = obj_tmpl_id[len(prefix):].strip('/.')
            return obj_tmpl_id
    return obj_tmpl_id


def get_uniform_view_pt_positions(num_angles: int=12, num_radii: int=5,
                                  min_radius: float=0.5, max_radius: float=3.0) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_angles)
    radii = np.linspace(min_radius, max_radius, num_radii, endpoint=True)
    rel_pos = np.zeros((num_radii * num_angles, 3), dtype=np.float32)
    rel_pos[:, 0] = np.outer(radii, np.cos(angles)).flatten()
    rel_pos[:, 2] = np.outer(radii, np.sin(angles)).flatten()
    return rel_pos


def get_random_view_pt_positions(num_pts: int,
                                 min_radius: float=0.5, max_radius: float=3.0) -> np.ndarray:
    angles = 2 * np.pi * np.random.random(num_pts)
    radii = (max_radius - min_radius) * np.random.random(num_pts) + min_radius
    rel_pos = np.zeros((num_pts, 3), dtype=np.float32)
    rel_pos[:, 0] = radii * np.cos(angles)
    rel_pos[:, 2] = radii * np.sin(angles)
    return rel_pos


def get_view_pt_pans(rel_pos: np.ndarray) -> np.ndarray:
    num_pts = rel_pos.shape[0]
    pan = np.arctan2(rel_pos[:, 0], rel_pos[:, 2])
    pan_q = np.zeros((num_pts, 4))
    pan_q[:, 0] = np.cos(0.5 * pan)
    pan_q[:, 2] = np.sin(0.5 * pan)
    return quaternion.from_float_array(pan_q)


def get_view_pt_tilts(rel_pos: np.ndarray) -> np.ndarray:
    num_pts = rel_pos.shape[0]
    hypot = np.hypot(rel_pos[:, 0], rel_pos[:, 2])
    tilt = np.arctan(-rel_pos[:, 1] / hypot)
    tilt_q = np.zeros((num_pts, 4))
    tilt_q[:, 0] = np.cos(0.5 * tilt)
    tilt_q[:, 1] = np.sin(0.5 * tilt)
    return quaternion.from_float_array(tilt_q)


def get_view_pt_rotations(rel_pos: np.ndarray) -> np.ndarray:
    pan_q = get_view_pt_pans(rel_pos)
    tilt_q = get_view_pt_tilts(rel_pos)
    return pan_q * tilt_q


def render_view_pts(sim: HabitatSim, abs_pos: np.ndarray, abs_rot: np.ndarray) \
                   -> Dict[str, np.ndarray]:
    # Get a copy to restore agent state at the end (with camera tilt when actionspace='v1')
    prv_s = sim.get_agent_state()
    s = sim.get_agent_state()
    observations = {uuid: [] for uuid in s.sensor_states}
    for pos, rot in zip(abs_pos, abs_rot):
        for uuid, sensor_s in s.sensor_states.items():
            sensor_s.position = pos
            sensor_s.rotation = rot
        sim.get_agent(0).set_state(s, False, False)
        obs = sim.sensor_suite.get_observations(sim.get_sensor_observations())
        for uuid in s.sensor_states:
            observations[uuid].append(obs[uuid])
    sim.get_agent(0).set_state(prv_s, False, False)
    return {uuid: np.stack(obs, 0) for uuid, obs in observations.items()}


class DebugMapRenderer:
    def __init__(self, outdir="out/spawned_objectnav_generator_debug/"):
        self.outdir = outdir
        self.create_outdir = True

    @staticmethod
    def put_text(disp, txt, emph=False):
        thick = 2 if emph else 1
        (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, thick)
        cv2.putText(disp, txt, (j - w // 2, i - b),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thick)

    def render(self, sim, distrib, goals, error):
        if self.create_outdir:
            os.makedirs(self.outdir, exist_ok=True)
            self.create_outdir = False
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
            bb = distrib.world_to_map(np.array(goal.pathfinder_targets))[..., [1, 0]]
            cv2.polylines(disp, bb.reshape(2, 4, 2), True, (255, 0, 255))

            for view_pt in goal.view_points:
                i, j = distrib.world_to_map(view_pt.position)
                cv2.circle(disp, (j, i), 3, (0, 255, 255), 2)

            put_text(disp, cat, goal is error.goal)

        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        cnt = 0
        outpath = os.path.join(self.outdir, f"DEBUG_render_{time_str}_{cnt}.png")
        while os.path.exists(outpath):
            cnt += 1
            outpath = os.path.join(self.outdir, f"DEBUG_render_{time_str}_{cnt}.png")
        cv2.imwrite(outpath, disp)
