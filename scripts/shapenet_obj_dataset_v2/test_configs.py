import sys
import os
import json
import multiprocessing as mp

import tqdm
import numpy as np
from habitat_sim import (Configuration, SimulatorConfiguration, AgentConfiguration,
                         CameraSensorSpec, Simulator)


INPUT_DIR = "data/object_datasets/shapenet/v2/.configs/"
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
OBJ_POS = np.array([0.0, 50.0, 0.0])
OBJ_SIZE = 0.6
OBJ_SIZE_REL_STD = 0.1


def make_sim_cfg():
    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    cam_spec = CameraSensorSpec()
    cam_spec.uuid = "rgb"
    cam_spec.position = [0.0, 0.5, 0.0]
    cam_spec.orientation = [-0.4636, 0.0, 0.0]
    cam_spec.hfov = 90
    cam_spec.resolution = [480, 640]
    ag_cfg = AgentConfiguration(sensor_specifications=[cam_spec])
    return Configuration(sim_cfg, [ag_cfg])


def worker(conn):
    cfg = make_sim_cfg()
    with Simulator(cfg) as sim:
        s = sim.get_agent(0).get_state()
        s.position = OBJ_POS + np.array([0.0, 0.0, 1.0])
        s.sensor_states = {}
        sim.get_agent(0).set_state(s)

        tmpl_mngr = sim.get_object_template_manager()
        templates_ids = tmpl_mngr.load_configs(INPUT_DIR)
        obj_mngr = sim.get_rigid_object_manager()
        conn.send(("SIM_LOADED", len(templates_ids)))
        while True:
            tmpl_index = conn.recv()
            if tmpl_index is None:
                break
            tmpl_id = templates_ids[tmpl_index]
            tmpl_path = tmpl_mngr.get_template_by_id(tmpl_id).handle
            tmpl_name = tmpl_path[tmpl_path.rfind('/') + 1:-len(".object_config.json")]

            obj = obj_mngr.add_object_by_template_id(tmpl_id)
            if obj is None:
                conn.send(("COULD_NOT_SPAWN", tmpl_name))
            else:
                size = obj.root_scene_node.cumulative_bb.size().max()
                if size > OBJ_SIZE * (1 + 3 * OBJ_SIZE_REL_STD):
                    conn.send(("OBJECT_TOO_BIG", tmpl_name))
                else:
                    obj.translation = OBJ_POS
                    view = sim.get_sensor_observations()['rgb']
                    if view[..., :3].any():
                        conn.send(("OBJECT_OK", tmpl_name))
                    else:
                        conn.send(("NOT_VISIBLE", tmpl_name))
                    obj_mngr.remove_object_by_handle(obj.handle)


def respawn_worker():
    conn, worker_conn = mp.Pipe()
    proc = mp.Process(target=worker, args=(worker_conn,))
    proc.start()
    msg, num_templates = conn.recv()
    if msg != "SIM_LOADED":
        raise ValueError(f"Unexpected message '{msg}' from worker.")
    return proc, conn, num_templates


def main():
    proc, conn, num_templates = respawn_worker()
    with tqdm.trange(num_templates, file=sys.stdout) as progress:
        for tmpl_index in progress:
            for _ in range(5):
                conn.send(tmpl_index)
                try:
                    msg, tmpl_name = conn.recv()
                    if msg == "OBJECT_OK":
                        progress.set_description(tmpl_name)
                        break
                    elif msg == "OBJECT_TOO_BIG":
                        progress.write(f"{tmpl_name} {msg}. Warning")
                        break
                    elif msg in ("COULD_NOT_LOAD", "COULD_NOT_SPAWN", "NOT_VISIBLE"):
                        progress.write(f"{tmpl_name} {msg}. Retrying")
                        conn.send(None)
                        proc.join()
                        proc, conn, num_reloaded_templates = respawn_worker()
                        assert num_reloaded_templates == num_templates
                    else:
                        raise ValueError(f"Unexpected message '{msg}' from worker.")
                except EOFError:
                    progress.write(f"CRASHED. Retrying")
                    conn.close()
                    proc.kill()
                    proc, conn, num_reloaded_templates = respawn_worker()
                    assert num_reloaded_templates == num_templates
            else:
                progress.write(f"FAILED. Aborting")
    conn.send(None)
    proc.join()


if __name__ == "__main__":
    main()
