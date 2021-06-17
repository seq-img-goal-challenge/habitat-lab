import os.path
import habitat
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
