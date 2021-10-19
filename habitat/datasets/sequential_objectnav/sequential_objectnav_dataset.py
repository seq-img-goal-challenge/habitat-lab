from typing import List, Dict, Set, Optional, Any
import os
import threading
import itertools
import gzip
import json
import random

from habitat.config.default import Config
from habitat.core.dataset import EpisodeIterator
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.datasets.spawned_objectnav.spawned_objectnav_dataset import (
        SpawnedObjectNavDatasetV0,
)
from habitat.datasets.spawned_objectnav.utils import (DEFAULT_SCENE_PATH_PREFIX,
                                                      DEFAULT_SCENE_PATH_EXT,
                                                      find_scene_file)
from habitat.tasks.nav.spawned_objectnav import ViewPoint, SpawnedObjectGoal
from habitat.tasks.sequential_nav.sequential_nav import SequentialDataset
from habitat.tasks.sequential_nav.sequential_objectnav import (SequentialObjectNavStep,
                                                               SequentialObjectNavEpisode)


CONTENT_DIR = "content"
ALL_SCENES_SELECTOR = "*"


@registry.register_dataset(name="SequentialObjectNav-v0")
class SequentialObjectNavDatasetV0(SpawnedObjectNavDatasetV0, SequentialDataset):
    config: Optional[Config] = None
    _loaded_episodes: List[SequentialObjectNavEpisode]
    _scene_content_paths: Optional[List[str]] = None
    _load_next: Optional[threading.Semaphore] = None
    _next_avail: Optional[threading.Semaphore] = None
    _load_thread: Optional[threading.Thread] = None
    _exit_thread: bool = False

    def __init__(self, config: Optional[Config]=None) -> None:
        self._loaded_episodes = []
        if config is None:
            return
        self.config = config
        data_path = config.DATA_PATH.format(split=config.SPLIT)
        content_path = os.path.join(os.path.dirname(data_path), CONTENT_DIR)
        use_all_scenes = ALL_SCENES_SELECTOR in config.CONTENT_SCENES
        ext_len = len(DEFAULT_SCENE_PATH_EXT)
        if os.path.isdir(content_path):
            self._scene_content_paths = [
                entry.path for entry in os.scandir(content_path)
                if use_all_scenes or entry.name[:-ext_len] in config.CONTENT_SCENES
            ]
            self._num_episodes = self._count_episodes_in_scenes()
            self._load_next = threading.Semaphore(1)
            self._next_avail = threading.Semaphore(0)
            self._load_thread = threading.Thread(target=self._load_func)
            self._load_thread.start()
        else:
            with gzip.open(data_path, 'rt') as f:
                self.from_json(f.read())

    def __del__(self):
        if self._scene_content_paths:
            self._exit_thread = True
            self._load_next.release()
            self._load_thread.join()

    def _load_func(self):
        load_cycle = itertools.cycle(self._scene_content_paths)
        logger.info("(loader) Data loading thread started")
        while True:
            self._load_next.acquire()
            if self._exit_thread:
                break
            path = next(load_cycle)
            logger.info(f"(loader) Loading next scene from '{path}'")
            self._loaded_episodes = []
            with gzip.open(path) as f:
                self.from_json(f.read())
            logger.info(f"(loader) Scene loaded from '{path}'")
            self._next_avail.release()
        logger.info("(loader) Data loading thread ended")

    def _count_episodes_in_scenes(self):
        logger.info("Counting total number of episodes in all selected scenes")
        cnt = 0
        for path in self._scene_content_paths:
            with gzip.open(path, 'rt') as f:
                raw = f.read()
            cnt += raw.count("episode_id")
        return cnt

    @property
    def num_episodes(self) -> int:
        if self._scene_content_paths is None:
            return super().num_episodes
        else:
            return self._num_episodes

    @property
    def episodes(self) -> List[SequentialObjectNavEpisode]:
        if self._scene_content_paths is None:
            return self._loaded_episodes
        else:
            self._next_avail.acquire()
            dummy = [self._loaded_episodes[0] for _ in range(self._num_episodes)] 
            self._next_avail.release()
            logger.warning("Returning a dummy list filled with first episode as 'dataset.episodes'... "
                           + "Please use 'get_episode_iterator()' or 'num_episodes' instead!")
            return dummy

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> EpisodeIterator:
        if self._scene_content_paths is None:
            return super().get_episode_iterator(*args, **kwargs)
        else:
            return SequentialObjectNavEpisodeIterator(self, *args, **kwargs)

    def get_objects_to_load(self, episode: Optional[SequentialObjectNavEpisode]=None) \
                           -> Set[str]:
        if episode is None:
            return {goal.object_template_id for episode in self.episodes
                                            for step in episode.steps
                                            for goal in step.goals}
        else:
            return {goal.object_template_id for step in episode.steps
                                            for goal in step.goals}

    def get_object_category_map(self) -> Dict[str, int]:
        return {step.object_category: step.object_category_index
                for episode in self.episodes for step in episode.steps}

    def get_max_object_category_index(self) -> int:
        return max(step.object_category_index
                   for episode in self.episodes for step in episode.steps)

    def _json_hook(self, raw_dict):
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

    def from_json(self, json_str: str, scenes_dir: Optional[str]=None) -> None:
        deserialized = json.loads(json_str, object_hook=self._json_hook)
        self._loaded_episodes.extend(deserialized["episodes"])


class SequentialObjectNavEpisodeIterator(EpisodeIterator):
    dataset: SequentialObjectNavDatasetV0
    _cumul_episodes: int

    def __init__(self, dataset: SequentialObjectNavDatasetV0,
                       *args: Any, **kwargs: Any) -> None:
        self.dataset = dataset
        dataset._next_avail.acquire()
        super().__init__(dataset._loaded_episodes, *args, **kwargs)
        dataset._load_next.release()
        self._cumul_episodes = len(self.episodes)
        if not self.group_by_scene:
            logger.warning("'group_by_scene' option ignored by episode iterator "
                           + "with dynamically loaded scene contents (always True)")
        if self.max_scene_repetition_episodes > 0:
            logger.warning("'max_scene_repeat_episodes' option ignored by episode iterator "
                           + "with dynamically loaded scene contents (not supported)")
        if self.max_scene_repetition_steps > 0:
            logger.warning("'max_scene_repeat_steps' option ignored by episode iterator "
                           + "with dynamically loaded scene contents (not supported)")

    def __next__(self):
        next_ep = next(self._iterator, None)
        if next_ep is None:
            if self.cycle or self._cumul_episodes < self.dataset.num_episodes:
                self.dataset._next_avail.acquire()
                self.episodes = self.dataset._loaded_episodes
                self._cumul_num_episodes += len(self.episodes)
                if self.shuffle:
                    random.shuffle(self.episodes)
                self._iterator = iter(self.episodes)
                self.dataset._load_next.release()
                next_ep = next(self._iterator)
            else:
                raise StopIteration
        return next_ep
