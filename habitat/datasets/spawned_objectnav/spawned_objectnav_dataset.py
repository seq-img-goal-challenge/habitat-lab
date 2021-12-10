from typing import List, Dict, Set, Optional, Any
import json
import os.path
import random
import itertools

import numpy as np

from habitat.core.dataset import EpisodeIterator
from habitat.core.registry import registry
from habitat.config.default import Config
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.spawned_objectnav.utils import (find_scene_file,
                                                      find_object_config_file,
                                                      strip_scene_id,
                                                      strip_object_template_id)
from habitat.tasks.nav.spawned_objectnav import SpawnedObjectGoal, SpawnedObjectNavEpisode


class SpawnedObjectNavEpisodeIterator(EpisodeIterator):
    group_by_object_subset: bool
    object_subsets_size: int
    _current_episode: Optional[SpawnedObjectNavEpisode] = None

    def __init__(self, group_by_object_subset: bool=True, object_subsets_size: int=2000,
                       *args: Any, **kwargs: Any) -> None:
        self.group_by_object_subset = group_by_object_subset
        self.object_subsets_size = object_subsets_size
        super().__init__(*args, **kwargs)
        if not self.group_by_scene and self.group_by_object_subset:
            self.episodes = self._group_object_subsets(self.episodes)
            self._iterator = iter(self.episodes)

    def __next__(self) -> SpawnedObjectNavEpisode:
        self._current_episode = super().__next__()
        return self._current_episode

    def _group_scenes(self, episodes: List[SpawnedObjectNavEpisode]) \
                     -> List[SpawnedObjectNavEpisode]:
        if self.group_by_object_subset:
            episodes = self._group_object_subsets(episodes)
        return super()._group_scenes(episodes)

    def _group_object_subsets(self, episodes: List[SpawnedObjectNavEpisode]) \
                             -> List[SpawnedObjectNavEpisode]:
        assert self.group_by_object_subset

        objects = {goal.object_template_id for ep in episodes for goal in ep.all_goals}
        objects = list(objects)
        random.shuffle(objects)
        object_subsets = [set(objects[b:b + self.object_subsets_size])
                          for b in range(0, len(objects), self.object_subsets_size)]
        subsets_map = {tmpl_id: subset_id for subset_id, subset in enumerate(object_subsets)
                       for tmpl_id in subset}
        return sorted(episodes, key=lambda ep: subsets_map[ep.goals[0].object_template_id])

    def get_object_subset(self) -> Set[str]:
        if self._current_episode is None:
            subset = set()
            scn_id = None
        else:
            subset = {goal.object_template_id for goal in self._current_episode.all_goals}
            scn_id = self._current_episode.scene_id
        episodes = []
        while len(subset) < self.object_subsets_size:
            ep = next(self._iterator, None)
            if ep is None or (scn_id is not None and ep.scene_id != scn_id):
                break
            subset |= {goal.object_template_id for goal in ep.all_goals}
            episodes.append(ep)
        self._iterator = itertools.chain(episodes, self._iterator)
        return subset


@registry.register_dataset(name="SpawnedObjectNav-v0")
class SpawnedObjectNavDatasetV0(PointNavDatasetV1):
    config: Config
    episodes: List[SpawnedObjectNavEpisode]
    _scn_prefix: Optional[str] = None
    _scn_ext: Optional[str] = None
    _obj_prefix: Optional[str] = None
    _obj_ext: Optional[str] = None
    _ep_iter: Optional[SpawnedObjectNavEpisodeIterator] = None

    def __init__(self, config: Optional[Config]=None):
        self.config = config
        super().__init__(config=config)

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

    def get_episode_iterator(self, *args: Any, **kwargs: Any) \
                            -> SpawnedObjectNavEpisodeIterator:
        if self._ep_iter is None:
            self._ep_iter = SpawnedObjectNavEpisodeIterator(episodes=self.episodes,
                                                            *args, **kwargs)
        return self._ep_iter
