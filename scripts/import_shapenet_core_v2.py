import os
import math
import subprocess
import json
import tqdm
import argparse


DEFAULT_ARGS = {"input_file": "ShapeNetCore.v2.zip",
                "output_dir": "data/object_datasets/shapenet_core_v2",
                "categories": ['*'],
                "display_cols": 5}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", "-i")
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--prompt", "-p", action='store_true')
    parser.add_argument("--categories", "-c", nargs='*')
    parser.add_argument("--display-cols", "-d", type=int)
    parser.add_argument("--verbose", "-v", action='count')
    parser.set_defaults(**DEFAULT_ARGS)
    return parser.parse_args()


ROOT = "ShapeNetCore.v2"
DEFAULT_EXCLUDED = ["*/screenshots/*", "*.binvox"]
OBJ_CFG_EXT = ".object_config.json"


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
                assert item["synsetId"] == children.pop()
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
    if '*' in desired_categories:
        desired_categories = set(categories)
    else:
        assert all(cat in categories for cat in desired_categories)
    return desired_categories


def check_existing_dirs(desired_categories, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing_dirs = {entry.name for entry in os.scandir(output_dir)}
    return desired_categories - existing_dirs


def build_include_filter(categories, to_extract):
    return ["{}/{}/*".format(ROOT, synset_id) for cat_id in to_extract
                                              for synset_id in categories[cat_id]]


def extract_files(categories, counts, to_extract, input_file):
    included = build_include_filter(categories, to_extract)
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
    try:
        with open(json_path) as f:
            centroid = json.load(f)["centroid"]
        os.remove(json_path)
    except FileNotFoundError as e:
        print("Warning! Model '{}' is not normalized as expected".format(model_entry.name))
        centroid = [0.0, 0.0, 0.0]

    rel_obj_path = os.path.join(model_entry.name, "models", "model_normalized.obj")
    with open(model_entry.path + OBJ_CFG_EXT, 'wt') as f:
        json.dump({"render_asset": rel_obj_path,
                   "up": [0.0, 1.0, 0.0],
                   "front": [0.0, 0.0, -1.0],
                   "COM": centroid,
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
        desired_cat = set(args.categories)
    desired_cat = check_desired_categories(categories, desired_cat)
    to_extract = check_existing_dirs(desired_cat, args.output_dir)
    extracted = extract_files(categories, counts, to_extract, args.input_file)
    move_files(categories, extracted, args.output_dir)
    cleanup()


if __name__ == "__main__":
    main(parse_args())
