import os
import math
import subprocess
import json
import tqdm
import argparse

import numpy as np


DEFAULT_ARGS = {"input_file": "ShapeNetCore.v2.zip",
                "output_dir": "data/object_datasets/shapenet_core_v2",
                "filter_files": None,
                "categories": "*",
                "exclude_categories": "",
                "display_cols": 5}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", "-i")
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--filter-files", "-f",
                        help="Comma-separated list of files to read filters from")
    parser.add_argument("--prompt", "-p", action='store_true')
    parser.add_argument("--categories", "-c",
                        help="Comma-separated list of categories to include")
    parser.add_argument("--exclude-categories", "-x",
                        help="Comma-separated list of categories to exclude")
    parser.add_argument("--display-cols", "-d", type=int)
    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


ROOT = "ShapeNetCore.v2"
DEFAULT_EXCLUDED = ["*/screenshots/*", "*.binvox"]
OBJ_CFG_EXT = ".object_config.json"
SIZE_REMAP = np.array([(0, 0), (1, 1), (4, 2), (12, 0.5), (60, 1)])
SIZE_VAR = 0.2


def parse_taxonomy(in_file, toplevel=True, children_to_toplevel=False):
    subprocess.run(["unzip", "-o", in_file, ROOT + "/taxonomy.json"],
                   stdout=subprocess.DEVNULL, check=True)
    taxonomy_path = os.path.join(ROOT, "taxonomy.json")
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)
    os.remove(taxonomy_path)
    children = []
    toplevel_parent = None
    categories = {}
    counts = {}
    for item in taxonomy:
        if toplevel:
            if children:
                if item["synsetId"] != children.pop():
                    raise ValueError("Unexpected synsetId encountered " \
                                     + "while exploring children in a branch of the taxonomy.")
                if children_to_toplevel:
                    categories[toplevel_parent].extend(item["children"])
            else:
                toplevel_parent = item["name"].split(',')[0]
                categories[toplevel_parent] = [item["synsetId"]]
                counts[toplevel_parent] = item["numInstances"]
            children.extend(reversed(item["children"]))
        else:
            name = item["name"].split(',')[0]
            categories[name] = [item["synsetId"]]
            counts[name] = item["numInstances"]
    return categories, counts


def read_filter_file(filter_file):
    included = set()
    excluded = set()
    with open(filter_file) as f:
        for l in f:
            try:
                op, cat = l.strip().split(" ", 1)
                if op == "+":
                    included.add(cat)
                    excluded.discard(cat)
                elif op == "-":
                    excluded.add(cat)
                    included.discard(cat)
                else:
                    continue
            except ValueError:
                continue
    return included, excluded


def apply_filters(desired_categories, exclude_categories, filter_files):
    desired_categories -= exclude_categories
    for path in filter_files.split(','):
        included, excluded = read_filter_file(path)
        desired_categories |= included
        desired_categories -= excluded
    return desired_categories


def prompt_user(counts, cols):
    counts = list(counts.items())
    max_len = max(len(cat) + len(str(c)) for cat, c in counts) + 6
    print("Available categories (w/. # models):")
    print("\n".join("  ".join(f" - {cat} ({c})".ljust(max_len)
                              for cat, c in counts[cols*i:cols*(i+1)])
                    for i in range(math.ceil(len(counts) / cols))))
    user_input = input("Desired categories? (comma-separated, default: '*')> ")
    return {cat.strip() for cat in user_input.split(',')} if user_input else {'*'}


def check_desired_categories(categories, desired_categories):
    available_categories = set(categories)
    if '*' in desired_categories:
        return available_categories
    else:
        unavailable = desired_categories - available_categories
        if unavailable:
            raise ValueError(f"Could not find categories {unavailable} in the taxonomy.")
        return desired_categories


def check_existing_dirs(desired_categories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing_dirs = {entry.name for entry in os.scandir(output_dir)}
    return desired_categories - existing_dirs


def build_zip_filter(categories, to_filter):
    return ["{}/{}/*".format(ROOT, synset_id) for cat_id in to_filter
                                              for synset_id in categories[cat_id]]


def extract_files(categories, counts, to_extract, input_file):
    included = build_zip_filter(categories, to_extract)
    with tqdm.tqdm(total=sum(counts[cat] for cat in to_extract)) as progress:
        with subprocess.Popen(["unzip", "-n", input_file] + included \
                              + ["-x"] + DEFAULT_EXCLUDED,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              encoding="utf-8") as proc:
            for l in proc.stdout:
                l = l.strip()
                if l.startswith("creating:") and len(l.split('/')) == 4:
                    progress.set_description(l.replace("creating:", "Extracting"))
                    progress.update()
    return to_extract


def make_object_config(model_entry, synset_id):
    json_path = os.path.join(model_entry.path, "models", "model_normalized.json")
    ox, oy, oz = 0.0, 0.0, 0.0
    scale = 1.0
    try:
        with open(json_path) as f:
            obj_data = json.load(f)
        os.remove(json_path)
        ox = 0.5 * (obj_data["min"][0] + obj_data["max"][0])
        oy = obj_data["min"][1]
        oz = 0.5 * (obj_data["min"][2] + obj_data["max"][2])
        size = max(up - low for up, low in zip(obj_data["max"], obj_data["min"]))
        scale = np.interp(size, SIZE_REMAP[:, 0], SIZE_REMAP[:, 1]) / size
        scale *= np.random.normal(1, SIZE_VAR)
    except FileNotFoundError as e:
        print("Warning! Model '{}' is not normalized as expected".format(model_entry.name))

    rel_obj_path = os.path.join(model_entry.name, "models", "model_normalized.obj")
    with open(model_entry.path + OBJ_CFG_EXT, 'wt') as f:
        json.dump({"render_asset": rel_obj_path,
                   "up": [0.0, 1.0, 0.0],
                   "front": [0.0, 0.0, -1.0],
                   "COM": [ox, oy, oz],
                   "scale": [scale, scale, scale],
                   "is_collidable": True,
                   "use_bounding_box_for_collision": True,
                   "requires_lighting": True,
                   "semantic_id": int(synset_id)}, f, indent=2)


def move_files(categories, extracted, output_dir):
    total = len(extracted)
    w_cat = max(len(cat) for cat in extracted)
    w_i = len(str(total))
    for i, cat in enumerate(extracted, start=1):
        out_base = os.path.join(output_dir, cat)
        for synset_id in categories[cat]:
            try:
                os.rename(os.path.join(ROOT, synset_id), out_base)
                break
            except FileNotFoundError:
                pass
        else:
            print("Warning! Cannot find extracted files for category '{}' ".format(cat) \
                  + "(synset IDs: {}), skipping...".format(categories[cat]))
            continue

        with os.scandir(out_base) as dir_iter:
            models = [entry for entry in dir_iter if entry.is_dir()]
        for entry in tqdm.tqdm(models, desc=f"Moving {cat: >{w_cat}} ({i: {w_i}d}/{total})"):
            make_object_config(entry, synset_id)


def cleanup():
    os.rmdir(ROOT)


def main(args):
    categories, counts = parse_taxonomy(args.input_file)
    if args.prompt:
        desired_cat = prompt_user(counts, args.display_cols)
    else:
        desired_cat = set(args.categories.split(','))
    desired_cat = check_desired_categories(categories, desired_cat)
    exclude_cat = set(args.exclude_categories.split(','))
    desired_cat = apply_filters(desired_cat, exclude_cat, args.filter_files)
    to_extract = check_existing_dirs(desired_cat, args.output_dir)
    extracted = extract_files(categories, counts, to_extract, args.input_file)
    move_files(categories, extracted, args.output_dir)
    cleanup()


if __name__ == "__main__":
    main(parse_args())
