from typing import Optional

import numpy as np
import scipy.ndimage as sp_img

from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.nav import NavMeshSettings

class SpawnPositionDistribution:
    _sim: HabitatSim
    _height: float
    _resolution: float
    _margin: float
    _inflate_radius: float
    _epsilon: float
    _origin: Optional[np.ndarray]
    _nav_mask: Optional[np.ndarray]
    _edges: Optional[np.ndarray]
    _distrib: Optional[np.ndarray]
    _cumul: Optional[np.ndarray]
    _conn_comp_masks: Optional[np.ndarray]
    _conn_comp_weights: Optional[np.ndarray]
    _num_conn_comp: int
    _rng: Optional[np.random.Generator]

    def __init__(self, sim: HabitatSim, height: Optional[float]=None,
                       resolution: float=0.02, margin: float=0.1,
                       inflate_radius: float=1.8, epsilon: float=0.001,
                       seed: Optional[int]=None) -> None:
        self._sim = sim
        self._height = sim.get_agent_state().position[1] if height is None else height
        self._resolution = resolution
        self._margin = margin
        self._inflate_radius = inflate_radius
        self._epsilon = epsilon

        self._origin = None
        self._nav_mask = None
        self._edges = None
        self._distrib = None
        self._cumul = None
        self._conn_comp_masks = None
        self._conn_comp_weights = None
        self._num_conn_comp = 0
        self._rng = None

        self._update_navmesh()
        if seed is not None:
            self.seed(seed)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, height: float) -> None:
        self._height = height
        self._update_edges()

    @property
    def resolution(self) -> float:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: float) -> None:
        self._resolution = resolution
        self._update_edges()

    @property
    def margin(self) -> float:
        return self._margin

    @margin.setter
    def margin(self, margin: float) -> None:
        self._margin = margin
        self._update_navmesh()

    @property
    def inflate_radius(self) -> float:
        return self._inflate_radius

    @inflate_radius.setter
    def inflate_radius(self, inflate_radius: float) -> None:
        self._inflate_radius = inflate_radius
        self._update_distrib()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
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
        edges = np.zeros_like(self._nav_mask)
        edges[:-1, :-1] = (self._nav_mask[:-1, :-1] != self._nav_mask[:-1, 1:]) \
                        | (self._nav_mask[:-1, :-1] != self._nav_mask[1:, :-1])
        self._edges = edges.astype(np.float32)
        self._update_distrib()
        self._reset_conn_comp()

    def _update_distrib(self) -> None:
        ker_sigma = 0.2 * self._inflate_radius / self._resolution
        distrib = sp_img.gaussian_filter(self._edges, ker_sigma, mode='constant', truncate=5)
        distrib += self._epsilon
        distrib[~self._nav_mask] = 0
        self._distrib = distrib
        self._cumul = self._distrib.flatten().cumsum()

    def _reset_conn_comp(self) -> None:
        self._conn_comp_masks = None
        self._conn_comp_weights = None
        self._num_conn_comp = 0

    def _update_conn_comp(self) -> None:
        labels, num_labels = sp_img.label(self._nav_mask, np.ones((3, 3)))
        masks = labels[:, :, np.newaxis] == 1 + np.arange(num_labels)
        areas = masks.sum(axis=(0, 1)) * self._resolution**2
        conn_comp_filter = areas > self._margin**2
        areas = areas[conn_comp_filter].astype(np.float32)
        self._conn_comp_masks = masks[:, :, conn_comp_filter]
        self._conn_comp_weights = areas / areas.sum()
        self._num_conn_comp, = areas.shape

    def get_nav_mask(self) -> np.ndarray:
        return self._nav_mask.copy()

    def get_origin(self) -> np.ndarray:
        return self._origin.copy()

    def get_map_edges(self) -> np.ndarray:
        return self._edges.copy()

    def get_spatial_distribution(self) -> np.ndarray:
        return self._distrib / self._cumul[-1]

    def get_num_connected_components(self) -> int:
        if self._conn_comp_masks is None:
            self._update_conn_comp()
        return self._num_conn_comp

    def get_connected_component(self, conn_comp_index: int) -> np.ndarray:
        if self._conn_comp_masks is None:
            self._update_conn_comp()
        return self._conn_comp_masks[:, :, conn_comp_index].copy()

    def seed(self, seed: Optional[int]=None) -> None:
        self._rng = np.random.default_rng(seed)

    def world_to_map(self, world_xyz: np.ndarray) -> np.ndarray:
        return ((world_xyz - self._origin) / self._resolution).astype(np.int64)[..., [2, 0]]

    def map_to_world(self, map_ij: np.ndarray,
                           map_j: Optional[np.ndarray]=None) -> np.ndarray:
        if map_j is None:
            map_i = map_ij[..., 0]
            map_j = map_ij[..., 1]
        else:
            map_i = map_ij
        world_xyz = np.tile(self._origin, map_i.shape + (1,))
        world_xyz[..., 0] += self._resolution * map_j
        world_xyz[..., 2] += self._resolution * map_i
        return world_xyz

    def sample(self, num_samples: int=1,
                     rng: Optional[np.random.Generator]=None) -> np.ndarray:
        if rng is None:
            if self._rng is None:
                self.seed()
            rng = self._rng
        u = self._cumul[-1] * rng.random(num_samples)
        flat_i = np.digitize(u, self._cumul)
        i, j = np.unravel_index(flat_i, self._distrib.shape)
        return self.map_to_world(i, j)

    def sample_from_connected_component(self, num_samples: int=1,
                                              conn_comp_index: Optional[int]=None,
                                              rng: Optional[np.random.Generator]=None) \
                                       -> np.ndarray:
        if rng is None:
            if self._rng is None:
                self.seed()
            rng = self._rng
        if self._conn_comp_masks is None:
            self._update_conn_comp()
        if conn_comp_index is None:
            conn_comp_index = rng.randint(self._num_conn_comp, p=self._conn_comp_weights)
        mask = self._conn_comp_masks[:, :, conn_comp_index]
        i, j = np.nonzero(mask)
        cumul = self._distrib[mask].cumsum()
        u = cumul[-1] * rng.random(num_samples)
        flat_i = np.digitize(u, cumul)
        return self.map_to_world(i[flat_i], j[flat_i])

    def sample_reachable_from_position(self, num_samples: int=1,
                                             position: Optional[np.ndarray]=None,
                                             rng: Optional[np.random.Generator]=None) \
                                      -> np.ndarray:
        if rng is None:
            if self._rng is None:
                self.seed()
            rng = self._rng
        if position is None:
            position = self._sim.get_agent_state().position
        if self._conn_comp_masks is None:
            self._update_conn_comp()
        i, j = self.world_to_map(position)
        conn_comp_index = np.argmax(self._conn_comp_masks[i, j])
        return self.sample_from_connected_component(num_samples, conn_comp_index, rng)
