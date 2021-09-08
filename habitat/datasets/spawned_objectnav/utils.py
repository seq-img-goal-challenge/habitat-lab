import os.path
from typing import Optional, Tuple, Dict

import numpy as np
import quaternion

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


def find_scene_file(scene_id):
    for prefix in ('.', DEFAULT_SCENE_PATH_PREFIX, HABITAT_SCENE_PATH_PREFIX):
        for ext in ('', DEFAULT_SCENE_PATH_EXT):
            path = os.path.join(prefix, scene_id + ext)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError("Could not find scene file '{}'".format(scene_id))


def find_object_config_file(tmpl_id):
    for prefix in ('.', DEFAULT_OBJECT_PATH_PREFIX, HABITAT_OBJECT_PATH_PREFIX):
        for ext in ('', DEFAULT_OBJECT_PATH_EXT):
            path = os.path.join(prefix, tmpl_id + ext)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError("Could not find object config file for '{}'".format(tmpl_id))


def strip_scene_id(scene_id):
    if scene_id.endswith(DEFAULT_SCENE_PATH_EXT):
        scene_id = scene_id[:-len(DEFAULT_SCENE_PATH_EXT)]
    for prefix in (DEFAULT_SCENE_PATH_PREFIX, HABITAT_SCENE_PATH_PREFIX):
        if scene_id.startswith(prefix):
            scene_id = scene_id[len(prefix):]
            return scene_id
    return scene_id


def strip_object_template_id(obj_tmpl_id):
    if obj_tmpl_id.endswith(DEFAULT_OBJECT_PATH_EXT):
        obj_tmpl_id = obj_tmpl_id[:-len(DEFAULT_OBJECT_PATH_EXT)]
    for prefix in (DEFAULT_OBJECT_PATH_PREFIX, HABITAT_OBJECT_PATH_PREFIX):
        if obj_tmpl_id.startswith(prefix):
            obj_tmpl_id = obj_tmpl_id[len(prefix):]
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


def get_view_pt_rotations(rel_pos: np.ndarray) -> np.ndarray:
    num_pts = rel_pos.shape[0]
    pan = np.arctan2(rel_pos[:, 0], rel_pos[:, 2])
    pan_q = np.zeros((num_pts, 4))
    pan_q[:, 0] = np.cos(0.5 * pan)
    pan_q[:, 2] = np.sin(0.5 * pan)
    pan_q = quaternion.from_float_array(pan_q)

    hypot = np.hypot(rel_pos[:, 0], rel_pos[:, 2])
    tilt = np.arctan(-rel_pos[:, 1] / hypot)
    tilt_q = np.zeros((num_pts, 4))
    tilt_q[:, 0] = np.cos(0.5 * tilt)
    tilt_q[:, 1] = np.sin(0.5 * tilt)
    tilt_q = quaternion.from_float_array(tilt_q)

    return pan_q * tilt_q


def render_view_pts(sim: HabitatSim, abs_pos: np.ndarray, abs_rot: np.ndarray) \
                   -> Dict[str, np.ndarray]:
    s = sim.get_agent_state()
    observations = {uuid: [] for uuid in s.sensor_states}
    for pos, rot in zip(abs_pos, abs_rot):
        for uuid, sensor_s in s.sensor_states.items():
            sensor_s.position = pos
            sensor_s.rotation = rot
        sim.get_agent(0).set_state(s, False, False)
        obs = sim.get_sensor_observations()
        for uuid in s.sensor_states:
            observations[uuid].append(obs[uuid])
    s.sensor_states = {}
    sim.get_agent(0).set_state(s)
    return {uuid: np.stack(obs, 0) for uuid, obs in observations.items()}
