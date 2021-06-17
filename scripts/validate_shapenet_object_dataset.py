import multiprocessing as mp
import os
import itertools
import logging

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
EXCLUDED = {
        #"chair/941720989a7af0248b500dd30d6dfd0", # wrong images location './', FIXED -> '../'
        #"chair/482afdc2ddc5546f764d42eddc669b23", # wrong images location './', FIXED -> '../'
        #"chair/b7a1ec97b8f85127493a4a2a112261d3", # cannot load texture '.jpg', FIXES -> '.JPG'
        #"chair/2b90701386f1813052db1dda4adf0a0c", # wrong images location './', FIXED -> '../'
        #"chair/7ad134826824de98d0bef5e87b92b95e", # wrong images location './', FIXED -> '../'
        #"telephone/89d70d3e0c97baaa859b0bef8825325f", # wrong texture '.png', FIXED -> '.jpg'
        "bus/2d44416a2f00fff08fd1fd158db1677c", # missing file 'texture4.jpg'
        "chair/2ae70fbab330779e3bff5a09107428a5", # empty texture path '../'
        "chair/a8c0ceb67971d0961b17743c18fb63dc", # empty texture path '../'
        "chair/f3c0ab68f3dab6071b17743c18fb63dc", # empty texture path '../'
        "chair/c70c1a6a0e795669f51f77a6d7299806", # empty texture path '../'
        "display/b0952767eeb21b88e2b075a80e28c81b", # Files 'texture0.jpg' and 'texture1.jpg' are empty
        "telephone/a4910da0271b6f213a7e932df8806f9e", # File 'texture1.jpg' is empty
}


def worker(conn):
    cfg = habitat.get_config("configs/tasks/pointnav_gibson.yaml",
                             ["SIMULATOR.SCENE", "data/scene_datasets/gibson/Denmark.glb"])
    sim_cfg = cfg.SIMULATOR
    sim = habitat.sims.make_sim(sim_cfg.TYPE, config=sim_cfg)
    tmpl_mngr = sim.get_object_template_manager()
    obj_pos = np.array([0.0, 50.0, 0.0])

    conn.send(None)

    while True:
        tmpl_id = conn.recv()
        if tmpl_id is None:
            break
        mngr_id, = tmpl_mngr.load_configs(tmpl_id)
        obj_id = sim.add_object(mngr_id)
        sim.set_translation(obj_pos, obj_id)

        views = []
        for _ in range(N_VIEWS):
            r = 2 * np.random.random() + 0.8
            a = 2 * np.pi * np.random.random()
            pos = obj_pos + [r * np.cos(a), -0.5, r * np.sin(a)]
            rot = [0, np.sin(0.25 * np.pi - 0.5 * a), 0, np.cos(0.225 * np.pi - 0.5 * a)]
            try:
                views.append(sim.get_observations_at(pos, rot)["rgb"][:, :, ::-1])
            except MemoryError:
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
    assert conn.recv() is None
    return p, conn


def main():
    p, conn = spawn_worker()

    pool, _ = create_object_pool(OBJECTS_DIR)
    total_count = sum(len(tmpl_pool) for tmpl_pool in pool.values())
    viz_count = 0
    with open("validate_shapenet_object_dataset.log", 'wt') as logf:
        with tqdm.tqdm(total=total_count) as progress:
            for cat, tmpl_pool in pool.items():
                for tmpl_id in tmpl_pool:
                    short_tmpl_id = tmpl_id[len(OBJECTS_DIR):-len(OBJECTS_EXT)]
                    progress.set_description(short_tmpl_id)
                    if short_tmpl_id not in EXCLUDED:
                        conn.send(tmpl_id)
                        for t in range(N_RETRIES):
                            try:
                                visible = conn.recv()
                                if visible:
                                    viz_count += 1
                                else:
                                    logf.write(f"{short_tmpl_id}\n")
                                break
                            except EOFError:
                                logging.warning("Simulator died, restarting.")
                                p.terminate()
                                conn.close()
                                p, conn = spawn_worker()
                                conn.send(tmpl_id)
                        else:
                            logging.warning("Simulator keeps dying, skipping this model...")
                    viz_rate = viz_count / (progress.n + 1)
                    progress.set_postfix_str(f"{viz_rate:.1%} visible")
                    progress.update()
    conn.send(None)
    p.join()


if __name__ == "__main__":
    main()
