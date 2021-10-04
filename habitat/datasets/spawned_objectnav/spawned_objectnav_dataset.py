from typing import List, Dict, Set, Optional, Any
import json

import numpy as np

from habitat.core.registry import registry
from habitat.config.default import Config
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.spawned_objectnav.utils import (find_scene_file,
                                                      find_object_config_file,
                                                      strip_scene_id,
                                                      strip_object_template_id)
from habitat.tasks.nav.spawned_objectnav import ViewPoint, SpawnedObjectGoal, \
                                                SpawnedObjectNavEpisode


@registry.register_dataset(name="SpawnedObjectNav-v0")
class SpawnedObjectNavDatasetV0(PointNavDatasetV1):
    episodes: List[SpawnedObjectNavEpisode]

    def __init__(self, config: Optional[Config]=None):
        self.config = config
        super().__init__(config=config)

    def get_objects_to_load(self, episode: Optional[SpawnedObjectNavEpisode]=None) -> Set[str]:
        if episode is None:
            return {goal.object_template_id for episode in self.episodes
                                            for goal in episode.goals}
        else:
            return {goal.object_template_id for goal in episode.goals}

    def get_object_category_map(self) -> Dict[str, int]:
        return {episode.object_category: episode.object_category_index
                for episode in self.episodes}

    def get_max_object_category_index(self) -> int:
        return max(episode.object_category_index
                   for episode in self.episodes)

    def _json_hook(self, raw_dict):
        if "episode_id" in raw_dict:
            scenes_dir = None if self.config is None else self.config.SCENES_DIR
            raw_dict["scene_id"] = find_scene_file(raw_dict["scene_id"], scenes_dir)
            return SpawnedObjectNavEpisode(**raw_dict)
        elif "object_template_id" in raw_dict:
            objects_dir = None if self.config is None else self.config.OBJECTS_DIR
            obj_tmpl_id = find_object_config_file(raw_dict["object_template_id"], objects_dir)
            raw_dict["object_template_id"] = obj_tmpl_id
            return SpawnedObjectGoal(**raw_dict)
        elif "iou" in raw_dict:
            return ViewPoint(**raw_dict)
        else:
            return raw_dict

    def from_json(self, json_str: str, scenes_dir: Optional[str]=None) -> None:
        deserialized = json.loads(json_str, object_hook=self._json_hook)
        self.episodes.extend(deserialized["episodes"])

    def _json_default(self, obj: object) -> Dict[str, Any]:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        s = obj.__getstate__() if hasattr(obj, "__getstate__") else obj.__dict__
        if "scene_id" in s:
            scenes_dir = None if self.config is None else self.config.SCENES_DIR
            s["scene_id"] = strip_scene_id(s["scene_id"], scenes_dir)
        elif "object_template_id" in s:
            objects_dir = None if self.config is None else self.config.OBJECTS_DIR
            s["object_template_id"] = strip_object_template_id(s["object_template_id"],
                                                               objects_dir)
        return s

    def to_json(self) -> str:
        return json.dumps({"episodes": self.episodes}, default=self._json_default)
