import os
import shutil
import json
import random
import re
from collections import defaultdict

from tqdm import tqdm


INPUT_DIR = "TEMP/ShapeNetCore.v2/"
LOG_PATH = "out/shapenet_obj_dataset_v2/test_all.out"
OUTPUT_DIR = "data/object_datasets/shapenet/v2"
INCLUDE_CAT = ("car", "train", "vessel", "airplane", "bus", "motorcycle",
               "guitar", "piano", "rifle", "pistol", "knife", "helmet")
NUM_OBJ_PER_CAT = 160
SEED = None


def highlight(cat):
    return "\x1b[32;1m" if cat in INCLUDE_CAT else "\x1b[31;2m"


def main():
    with open(os.path.join(INPUT_DIR, "taxonomy.json")) as f:
        cat_map = {elem["synsetId"]: elem["name"].split(',')[0] for elem in json.load(f)}
    available = {entry.name for entry in os.scandir(INPUT_DIR) if entry.is_dir()}
    cat_map = {synset_id: cat_map[synset_id] for synset_id in available}
    num_cat = len(INCLUDE_CAT)
    synset_strings = [f"{highlight(v)}{k: >15}: {v: <20}\x1b[0m"
                      for k, v in cat_map.items()]
    print('-'*151)
    print("\n".join("|".join(synset_str for synset_str in synset_strings[i:i+4])
                    for i in range(0, len(synset_strings), 4)))

    print('-'*151)
    print(f"++ {num_cat} categories \x1b[32;1mselected\x1b[0m. ++")
    print(f"-- {len(available) - num_cat} categories \x1b[31;2mexcluded\x1b[0m. --")
    print('-'*151)

    pattern = re.compile(r".*/(?P<synset_id>.*)/(?P<obj_id>.*) \((?P<cat_name>.*)\) FAILED. Aborting")
    failures = defaultdict(list)
    with open(LOG_PATH) as f:
        for l in f:
            if match := pattern.match(l):
                failures[match["synset_id"]].append(match["obj_id"])
    num_fail = sum(len(cat_fail) for cat_fail in failures.values())
    print(f"-- {num_fail} invalid objects to be \x1b[31;2mavoided\x1b[0m. --")


    synset_map = {cat_name: synset_id for synset_id, cat_name in cat_map.items()}
    seed = random.randint(1000, 9999) if SEED is None else SEED
    print(f"Seeding RNG -> {seed}")
    random.seed(seed)
    for k, cat_name in enumerate(INCLUDE_CAT, start=1):
        os.makedirs(os.path.join(OUTPUT_DIR, cat_name))
        synset_id = synset_map[cat_name]
        obj_ids = sorted(entry.name for entry in os.scandir(os.path.join(INPUT_DIR, synset_id))
                         if entry.name not in failures[synset_id])
        random.shuffle(obj_ids)
        obj_selection = obj_ids[:NUM_OBJ_PER_CAT]
        for obj_id in tqdm(obj_selection, desc=f"{cat_name: >15} ({k: >2}/{num_cat})",
                           position=0, leave=False):
            shutil.copytree(os.path.join(INPUT_DIR, synset_id, obj_id),
                            os.path.join(OUTPUT_DIR, cat_name, obj_id))


if __name__ == "__main__":
    main()
