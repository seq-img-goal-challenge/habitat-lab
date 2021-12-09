from typing import List, Dict, Set, Optional, Any
import os.path
import json

from habitat.core.registry import registry
from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset import (
    SpawnedObjectNavDatasetV0
)
from habitat.datasets.spawned_objectnav.utils import find_scene_file
from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
from habitat.tasks.sequential_nav.sequential_objectnav import (SequentialObjectNavStep,
                                                               SequentialObjectNavEpisode)


@registry.register_dataset(name="SequentialObjectNav-v0")
class SequentialObjectNavDatasetV0(SpawnedObjectNavDatasetV0, SequentialDataset):
    episodes: List[SequentialObjectNavEpisode]

    def get_object_category_map(self) -> Dict[str, int]:
        return {step.object_category: step.object_category_index
                for episode in self.episodes for step in episode.steps}

    def get_max_object_category_index(self) -> int:
        return max(step.object_category_index
                   for episode in self.episodes for step in episode.steps)

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
            return SequentialObjectNavEpisode(**raw_dict)
        elif "object_category" in raw_dict:
            return SequentialObjectNavStep(**raw_dict)
        else:
            return super()._json_hook(raw_dict)
