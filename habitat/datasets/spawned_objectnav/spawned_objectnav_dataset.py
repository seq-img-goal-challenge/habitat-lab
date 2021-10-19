from typing import List, Dict, Set, Optional, Any
import json
import os.path

import numpy as np

from habitat.core.registry import registry
from habitat.config.default import Config
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.spawned_objectnav.utils import (find_scene_file,
                                                      find_object_config_file,
                                                      strip_scene_id,
                                                      strip_object_template_id)
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode

@registry.register_dataset(name="SpawnedObjectNav-v0")
class SpawnedObjectNavDatasetV0(PointNavDatasetV1):
    config: Config
    episodes: List[SpawnedObjectNavEpisode]
    _scn_prefix: Optional[str] = None
    _scn_ext: Optional[str] = None
    _obj_prefix: Optional[str] = None
    _obj_ext: Optional[str] = None

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

    def _json_hook(self, raw_dict: Dict[str, Any]) -> Any:
        if "episode_id" in raw_dict:
            if self._scn_prefix is None:
                scenes_dir = None if self.config is None else self.config.SCENES_DIR
                raw_dict["scene_id"], self._scn_prefix, self._scn_ext = find_scene_file(
                    raw_dict["scene_id"], scenes_dir
                )
            else:
                raw_dict["scene_id"] = os.path.join(self._scn_prefix,
                                                    raw_dict["scene_id"] + self._scn_ext)
            return SpawnedObjectNavEpisode(**raw_dict)
        elif "object_template_id" in raw_dict:
            objects_dir = None if self.config is None else self.config.OBJECTS_DIR
            if self._obj_prefix is None:
                obj_tmpl_id, self._obj_prefix, self._obj_ext = find_object_config_file(
                    raw_dict["object_template_id"], objects_dir
                )
            else:
                obj_tmpl_id = os.path.join(self._obj_prefix,
                                           raw_dict["object_template_id"] + self._obj_ext)
            raw_dict["object_template_id"] = obj_tmpl_id
            view_pts_indices = np.array(raw_dict["valid_view_points_indices"])
            raw_dict["valid_view_points_indices"] = view_pts_indices.astype(np.uint8)
            view_pts_ious = np.array(raw_dict["valid_view_points_ious"])
            raw_dict["valid_view_points_ious"] = view_pts_ious.astype(np.float32)
            return SpawnedObjectGoal(**raw_dict)
        else:
            return raw_dict

    def from_json(self, json_str: str, scenes_dir: Optional[str]=None) -> None:
        deserialized = json.loads(json_str, object_hook=self._json_hook)
        self.episodes.extend(deserialized["episodes"])

    def _json_default(self, obj: Any) -> Dict[str, Any]:
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
