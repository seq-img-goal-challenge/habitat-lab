import multiprocessing as mp
import os
import itertools
import logging
import enum

import numpy as np
import quaternion
import cv2
import tqdm

os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"

import habitat
from habitat.datasets.spawned_objectnav.spawned_objectnav_generator import create_object_pool
habitat.logger.setLevel(logging.ERROR)


OBJECTS_DIR = "data/object_datasets/shapenet_core_v2/"
OBJECTS_EXT = ".object_config.json"
N_VIEWS = 8
N_RETRIES = 2
TIMEOUT=4.0
LOG_PATH = "validate_shapenet_object_dataset.log"


class WORKER_MESSAGES(enum.Enum):
    READY = enum.auto()
    END = enum.auto()


def worker(conn):
    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml",
                             ["SIMULATOR.SCENE", "data/scene_datasets/gibson/Denmark.glb"])
    sim_cfg = cfg.SIMULATOR
    sim = habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg)
    tmpl_mngr = sim.get_object_template_manager()
    obj_pos = np.array([0.0, 50.0, 0.0])

    conn.send(WORKER_MESSAGES.READY)

    while True:
        tmpl_id = conn.recv()
        if tmpl_id is WORKER_MESSAGES.END:
            break
        try:
            mngr_id, = tmpl_mngr.load_configs(tmpl_id)
        except MemoryError:
            logging.info("Simulator failed to load object config.")
            break
        try:
            obj_id = sim.add_object(mngr_id)
        except MemoryError:
            logging.info("Simulator failed to spawn object in scene.")
            break
        sim.set_translation(obj_pos, obj_id)

        views = []
        try:
            for _ in range(N_VIEWS):
                r = 2 * np.random.random() + 0.8
                a = 2 * np.pi * np.random.random()
                pos = obj_pos + [r * np.cos(a), -0.5, r * np.sin(a)]
                rot = [0, np.sin(0.25 * np.pi - 0.5 * a), 0, np.cos(0.225 * np.pi - 0.5 * a)]
                views.append(sim.get_observations_at(pos, rot)["rgb"][:, :, ::-1])
        except MemoryError:
            logging.info("Simulator failed to generate a view.")
            break
        views = np.concatenate(views, 1)
        conn.send(views.max() > 0)
        sim.remove_object(obj_id)
        tmpl_mngr.remove_template_by_ID(mngr_id)
    conn.close()


def spawn_worker():
    conn, child_conn = mp.Pipe()
    p = mp.Process(target=worker, args=(child_conn,))
    p.start()
    assert conn.recv() is WORKER_MESSAGES.READY
    return p, conn


def main():
    p, conn = spawn_worker()

    pool, _ = create_object_pool(OBJECTS_DIR)
    total_count = sum(len(tmpl_pool) for tmpl_pool in pool.values())
    viz_count = 0
    with open(LOG_PATH, 'wt') as logf:
        with tqdm.tqdm(total=total_count) as progress:
            for cat, tmpl_pool in pool.items():
                for tmpl_id in tmpl_pool:
                    short_tmpl_id = tmpl_id[len(OBJECTS_DIR):-len(OBJECTS_EXT)]
                    progress.set_description(short_tmpl_id)
                    conn.send(tmpl_id)
                    for t in range(N_RETRIES):
                        try:
                            visible = conn.recv()
                            if visible:
                                viz_count += 1
                            else:
                                logf.write(f"{short_tmpl_id}\n")
                            break
                        except (EOFError, ConnectionResetError):
                            logging.warning("Simulator died, restarting.")
                            p.terminate()
                            conn.close()
                            p, conn = spawn_worker()
                            conn.send(tmpl_id)
                    else:
                        logging.warning("Simulator keeps dying, skipping this model.")
                    viz_rate = viz_count / (progress.n + 1)
                    progress.set_postfix_str(f"{viz_rate:.1%} visible")
                    progress.update()
    conn.send(WORKER_MESSAGES.END)
    p.join()


if __name__ == "__main__":
    main()
