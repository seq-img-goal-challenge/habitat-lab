from typing import List, Dict, Optional
import os.path
import json

from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1, \
                                                       DEFAULT_SCENE_PATH_PREFIX
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode


DEFAULT_OBJECT_PATH_PREFIX = "data/object_datasets/"
DEFAULT_OBJECT_PATH_EXT = ".object_config.json"


@registry.register_dataset(name="SpawnedObjectNav-v0")
class SpawnedObjectNavDatasetV0(PointNavDatasetV1):
    episodes: List[SpawnedObjectNavEpisode]

    @staticmethod
    def _json_hook(raw_dict):
        if "episode_id" in raw_dict:
            if raw_dict["scene_id"].startswith(DEFAULT_SCENE_PATH_PREFIX):
                raw_dict["scene_id"] = raw_dict["scene_id"][len(DEFAULT_SCENE_PATH_PREFIX):]
            return SpawnedObjectNavEpisode(**raw_dict)
        elif "object_template_id" in raw_dict:
            return SpawnedObjectGoal(**raw_dict)
        else:
            return raw_dict

    @staticmethod
    def _find_object_config_file(tmpl_id):
        for prefix in ('.', DEFAULT_OBJECT_PATH_PREFIX):
            for ext in ('', DEFAULT_OBJECT_PATH_EXT):
                path = os.path.join(prefix, tmpl_id + ext)
                if os.path.isfile(path):
                    return path
        raise FileNotFoundError("Could not find object config file for '{}'".format(tmpl_id))

    def get_objects_to_load(self,
                            episode: Optional[SpawnedObjectNavEpisode]=None) -> List[str]:
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

    def from_json(self, json_str: str, scenes_dir: Optional[str]=None) -> None:
        deserialized = json.loads(json_str, object_hook=self._json_hook)
        self.episodes = deserialized["episodes"]
