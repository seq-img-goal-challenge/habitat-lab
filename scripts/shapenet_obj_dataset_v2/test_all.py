import sys
import os
import json
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
from habitat_sim import (Configuration, SimulatorConfiguration, AgentConfiguration,
                         CameraSensorSpec, Simulator)


INPUT_DIR = "TEMP/ShapeNetCore.v2/"
OUTPUT_DIR = "out/shapenet_obj_dataset_v2/"
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
OBJ_POS = np.array([0.0, 50.0, 0.0])


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
        obj_mngr = sim.get_rigid_object_manager()
        conn.send("SIM_LOADED")
        while True:
            cat_name, obj_path = conn.recv()
            if cat_name == "END":
                break
            obj_name = os.path.basename(obj_path)

            tmpl = tmpl_mngr.create_new_template(f"{cat_name}_{obj_name}")
            tmpl.render_asset_handle = os.path.join(obj_path, "models", "model_normalized.obj")
            tmpl.bounding_box_collisions = True
            tmpl.shader_type = 0
            tmpl.requires_lighting = False
            tmpl_id = tmpl_mngr.register_template(tmpl)
            if tmpl_id < 0:
                conn.send("COULD_NOT_LOAD")
                continue

            obj = obj_mngr.add_object_by_template_handle(tmpl.handle)
            if obj is None:
                conn.send("COULD_NOT_SPAWN")
            else:
                obj.translation = OBJ_POS
                view = sim.get_sensor_observations()['rgb']
                if view[..., :3].any():
                    conn.send("OBJECT_OK")
                else:
                    conn.send("NOT_VISIBLE")
                obj_mngr.remove_object_by_handle(obj.handle)
            tmpl_mngr.remove_template_by_handle(tmpl.handle)


def respawn_worker():
    conn, worker_conn = mp.Pipe()
    proc = mp.Process(target=worker, args=(worker_conn,))
    proc.start()
    msg = conn.recv()
    if msg != "SIM_LOADED":
        raise ValueError(f"Unexpected message '{msg}' from worker.")
    return proc, conn


def main():
    with open(os.path.join(INPUT_DIR, "taxonomy.json")) as f:
        cat_map = {elem["synsetId"]: elem["name"].split(',')[0] for elem in json.load(f)}

    objects = {cat_map[cat_entry.name]: [obj_entry.path
                                         for obj_entry in os.scandir(cat_entry.path)]
               for cat_entry in os.scandir(INPUT_DIR) if cat_entry.is_dir()}
    total = sum(len(cat_objects) for cat_objects in objects.values())

    my_error_log = open(os.path.join(OUTPUT_DIR, "test_all.out"), 'wt')
    proc, conn = respawn_worker()
    with tqdm(total=total, file=sys.stdout) as progress:
        for cat_name, cat_objects in objects.items():
            for obj_path in cat_objects:
                for _ in range(5):
                    conn.send((cat_name, obj_path))
                    try:
                        msg = conn.recv()
                        if msg == "OBJECT_OK":
                            break
                        elif msg in ("COULD_NOT_LOAD", "COULD_NOT_SPAWN", "NOT_VISIBLE"):
                            progress.write(f"{obj_path} ({cat_name}) {msg}. Retrying")
                            my_error_log.write(f"{obj_path} ({cat_name}) {msg}. Retrying\n")
                            my_error_log.flush()
                            conn.send(("END", None))
                            proc.join()
                            proc, conn = respawn_worker()
                        else:
                            raise ValueError(f"Unexpected message '{msg}' from worker.")
                    except EOFError:
                        progress.write(f"{obj_path} ({cat_name}) CRASHED. Retrying")
                        my_error_log.write(f"{obj_path} ({cat_name}) CRASHED. Retrying\n")
                        my_error_log.flush()
                        conn.close()
                        proc.kill()
                        proc, conn = respawn_worker()
                else:
                    progress.write(f"{obj_path} ({cat_name}) FAILED. Aborting")
                    my_error_log.write(f"{obj_path} ({cat_name}) FAILED. Aborting\n")
                    my_error_log.flush()
                progress.set_description(obj_path)
                progress.update()
    conn.send(("END", None))
    proc.join()
    my_error_log.close()


if __name__ == "__main__":
    main()
