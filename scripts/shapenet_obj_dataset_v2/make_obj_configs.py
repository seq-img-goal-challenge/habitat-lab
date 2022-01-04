import sys
import os
import json
import multiprocessing as mp
import random

from tqdm import tqdm
import numpy as np
import quaternion
from habitat_sim import (Configuration, SimulatorConfiguration, AgentConfiguration,
                         CameraSensorSpec, Simulator)


TAXONOMY_PATH = "TEMP/ShapeNetCore.v2/taxonomy.json"
INPUT_DIR = "data/object_datasets/shapenet/v2/"
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
OBJ_POS = np.array([0.0, 50.0, 0.0])
OBJ_SIZE = 0.6
OBJ_SIZE_REL_STD = 0.1
SEED = 2315918


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
    with open(TAXONOMY_PATH) as f:
        synset_map = {elem["name"].split(',')[0]: elem["synsetId"] for elem in json.load(f)}

    with Simulator(cfg) as sim:
        s = sim.get_agent(0).get_state()
        s.position = OBJ_POS + np.array([0.0, 0.0, 1.0])
        s.sensor_states = {}
        sim.get_agent(0).set_state(s)
        view = sim.get_sensor_observations()['rgb']

        tmpl_mngr = sim.get_object_template_manager()
        obj_mngr = sim.get_rigid_object_manager()
        conn.send("SIM_LOADED")
        while True:
            cat_name, obj_path = conn.recv()
            if cat_name == "END":
                break
            obj_id = os.path.basename(obj_path)

            tmpl = tmpl_mngr.create_new_template(f"{cat_name}_{obj_id}")
            tmpl.render_asset_handle = os.path.join(obj_path, "models", "model_normalized.obj")
            tmpl.bounding_box_collisions = True
            tmpl.shader_type = 0
            tmpl.requires_lighting = True
            tmpl_id = tmpl_mngr.register_template(tmpl)
            if tmpl_id < 0:
                conn.send("COULD_NOT_LOAD")
                continue

            obj = obj_mngr.add_object_by_template_handle(tmpl.handle)
            if obj is None:
                conn.send("COULD_NOT_SPAWN")
            else:
                size = obj.root_scene_node.cumulative_bb.size().max()
                random.seed(SEED ^ hash(obj_path))
                scale = OBJ_SIZE / size * random.gauss(1, OBJ_SIZE_REL_STD)

                with open(os.path.join(INPUT_DIR, ".configs",
                                       f"{cat_name}_{obj_id}.object_config.json"), 'wt') as f:
                    json.dump({"render_asset": os.path.join("..", cat_name, obj_id, "models", "model_normalized.obj"),
                               "shader_type": "material",
                               "requires_lighting": True,
                               "use_bouding_box_for_collision": True,
                               "scale": [scale, scale, scale],
                               "semantic_id": synset_map[cat_name]}, f, indent=2)

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
    os.makedirs(os.path.join(INPUT_DIR, ".configs"))

    objects = {cat_entry.name: [obj_entry.path for obj_entry in os.scandir(cat_entry.path)]
               for cat_entry in os.scandir(INPUT_DIR) if cat_entry.is_dir()}
    total = sum(len(cat_objects) for cat_objects in objects.values())

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
                            conn.send(("END", None))
                            proc.join()
                            proc, conn = respawn_worker()
                        else:
                            raise ValueError(f"Unexpected message '{msg}' from worker.")
                    except EOFError:
                        progress.write(f"{obj_path} ({cat_name}) CRASHED. Retrying")
                        conn.close()
                        proc.kill()
                        proc, conn = respawn_worker()
                else:
                    progress.write(f"{obj_path} ({cat_name}) FAILED. Aborting")
                progress.set_description(obj_path)
                progress.update()
    conn.send(("END", None))
    proc.join()


if __name__ == "__main__":
    main()

