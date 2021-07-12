from typing import Optional

import numpy as np
import scipy.ndimage as sp_img

from habitat.core.simulator import Simulator
from habitat_sim.nav import NavMeshSettings

class SpawnPositionDistribution:
    _sim: Simulator
    _height: float
    _resolution: float
    _margin: float
    _inflate_radius: float
    _epsilon: float
    _nav_mask: Optional[np.ndarray]
    _conn_comp_masks: Optional[np.ndarray]
    _conn_comp_weights: Optional[np.ndarray]
    _num_conn_comp: int
    _origin: Optional[np.ndarray]
    _edges: Optional[np.ndarray]
    _distrib: Optional[np.ndarray]
    _cumul: Optional[np.ndarray]
    _rng: Optional[np.random.Generator]

    def __init__(self, sim: Simulator, height: Optional[float]=None,
                       resolution: float=0.02, margin: float=0.1,
                       inflate_radius: float=1.8, epsilon: float=0.001,
                       seed: Optional[int]=None) -> None:
        self._sim = sim
        self._height = sim.get_agent_state().position[1] if height is None else height
        self._resolution = resolution
        self._margin = margin
        self._inflate_radius = inflate_radius
        self._epsilon = epsilon

        self._nav_mask = None
        self._conn_comp_masks = None
        self._conn_comp_weights = None
        self._num_conn_comp = 0
        self._origin = None
        self._edges = None
        self._distrib = None
        self._cumul = None
        self._rng = None

        self._update_navmesh()
        if seed is not None:
            self.seed(seed)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def set_height(self, height: float) -> None:
        self._height = height
        self._update_edges()

    @property
    def resolution(self) -> float:
        return self._resolution

    @resolution.setter
    def set_resolution(self, resolution: float) -> None:
        self._resolution = resolution
        self._update_edges()

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def set_margin(self, margin: float) -> None:
        self._margin = margin
        self._update_navmesh()

    @property
    def inflate_radius(self) -> float:
        return self._inflate_radius

    @inflate_radius.setter
    def set_inflate_radius(self, inflate_radius: float) -> None:
        self._inflate_radius = inflate_radius
        self._update_distrib()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def set_epsilon(self, epsilon: float) -> None:
        self._epsilon = epsilon
        self._update_distrib()

    def _update_navmesh(self) -> None:
        settings = NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = self._margin
        settings.agent_height = self._margin
        self._sim.recompute_navmesh(self._sim.pathfinder, settings)
        self._origin, _ = self._sim.pathfinder.get_bounds()
        self._origin[1] = self._height
        self._update_edges()

    def _update_edges(self) -> None:
        self._nav_mask = self._sim.pathfinder.get_topdown_view(self._resolution, self._height)
        edges = (self._nav_mask[:-1, :-1] != self._nav_mask[:-1, 1:]) \
                | (self._nav_mask[:-1, :-1] != self._nav_mask[1:, :-1])
        self._edges = edges.astype(np.float32)
        self._update_conn_comp()
        self._update_distrib()

    def _update_conn_comp(self) -> None:
        labels, num_labels = sp_img.label(self._nav_mask[:-1, :-1], np.ones((3, 3)))
        masks = labels[:, :, np.newaxis] == 1 + np.arange(num_labels)
        areas = masks.sum(axis=(0, 1)) * self._resolution**2
        # TODO: refine min area for components filtering
        conn_comp_filter = areas > self._margin**2
        areas = areas[conn_comp_filter]
        self._conn_comp_masks = masks[:, :, conn_comp_filter]
        self._conn_comp_weights = areas / areas.sum()
        self._num_conn_comp, = areas.shape

    def _update_distrib(self) -> None:
        ker_sigma = 0.2 * self._inflate_radius / self._resolution
        distrib = sp_img.gaussian_filter(self._edges, ker_sigma, mode='constant', truncate=5)
        distrib += self._epsilon
        distrib[~self._nav_mask[:-1, :-1]] = 0
        conn_comp = np.broadcast_to(distrib[:, :, np.newaxis],
                                    self._conn_comp_masks.shape)[self._conn_comp_masks]
        self._distrib = distrib
        self._cumul = conn_comp.reshape(-1, self._num_conn_comp).cumsum(axis=0)

    def get_origin(self) -> np.ndarray:
        return self._origin.copy()

    def get_map_edges(self) -> np.ndarray:
        return self._edges.copy()

    def get_spatial_distribution(self) -> np.ndarray:
        return self._distrib.sum(axis=-1) / self._distrib.sum()

    def seed(self, seed: Optional[int]=None) -> None:
        self._rng = np.random.default_rng(seed)

    def world_to_map(self, world_xyz: np.ndarray) -> np.ndarray:
        return ((world_xyz - self._origin) / self._resolution).astype(np.int64)[:, [2, 0]]

    def map_to_world(self, map_ij: np.ndarray,
                           map_j: Optional[np.ndarray]=None) -> np.ndarray:
        if map_j is None:
            map_i = map_ij[:, 0]
            map_j = map_ij[:, 1]
        else:
            map_i = map_ij
        world_xyz = np.tile(self._origin, (map_i.shape[0], 1))
        world_xyz[:, 0] += self._resolution * map_j
        world_xyz[:, 2] += self._resolution * map_i
        return world_xyz

    def sample(self, num_samples: int=1, conn_comp: Optional[int]=None,
                     rng: Optional[np.random.Generator]=None) -> np.ndarray:
        if rng is None:
            if self._rng is None:
                self.seed()
            rng = self._rng
        if conn_comp is None:
            conn_comp = rng.choice(self._num_conn_comp, p=self._conn_comp_weights)
        cumul = self._cumul[:, conn_comp]
        u = cumul[-1] * rng.random(num_samples)
        flat_i = np.digitize(u, cumul)
        i, j = np.unravel_index(flat_i, self._distrib.shape)
        return self.map_to_world(i, j)
