from typing import List, Dict, Set, Optional
import os.path
import json

import habitat
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.spawned_objectnav.utils import find_scene_file, find_object_config_file
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode


@registry.register_dataset(name="SpawnedObjectNav-v0")
class SpawnedObjectNavDatasetV0(PointNavDatasetV1):
    episodes: List[SpawnedObjectNavEpisode]

    @staticmethod
    def _json_hook(raw_dict):
        if "episode_id" in raw_dict:
            raw_dict["scene_id"] = find_scene_file(raw_dict["scene_id"])
            return SpawnedObjectNavEpisode(**raw_dict)
        elif "object_template_id" in raw_dict:
            obj_tmpl_id = find_object_config_file(raw_dict["object_template_id"])
            raw_dict["object_template_id"] = obj_tmpl_id
            return SpawnedObjectGoal(**raw_dict)
        else:
            return raw_dict

    def get_objects_to_load(self,
                            episode: Optional[SpawnedObjectNavEpisode]=None) -> Set[str]:
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
