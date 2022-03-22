from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from habitat_baselines.common.tensor_dict import TensorDict


def _ego_to_episodic_coords(
    observations: TensorDict,
    ego_coords: torch.FloatTensor
) -> torch.FloatTensor:
    r"""Transforms points given by their coordinates in the ego frame
    (ie. relative to the current position and orientation of the agent)
    to coordinates in the episodic frame
    (ie. relative to the initial position and orientation of the agent
    at the beginning of the episode)

    :param observations: Batch of observations as a dict containing:
        - 'compass': Current orientation (yaw, in radians)
            of the agent relative to its starting orientation
            as a :math:`B\times 1` tensor, where :math:`B > 0` is the batch size
        - 'gps': Current position (x=forward, y=right if 2D) or (x=right, y=up, z=back if 3D)
            of the agent relative to its starting position and orientation
            as a :math:`B\times D` tensor, where :math:`D\in\{2, 3\}` is the spatial dimension
    :type observations: habitat_baselines.TensorDict
    :param ego_coords: Ego coordinates (2D or 3D) of points to be transformed
        relative to the current position and orientation of the agent
        as a :math:`B\times\dots\times D` tensor
    :type ego_coords: torch.FloatTensor
    :returns: Episodic coordinates (2D or 3D) of every point to be transformed
        relative to the initial position and orientation of the agent
        at the beginning of the episode as a :math:`B\times\dots\times D` tensor
    :rtype: torch.FloatTensor
    """
    batch_size, *size, dim = ego_coords.size()
    a = observations["compass"].unsqueeze(2)
    cosa = torch.cos(a)
    sina = torch.sin(a)
    if dim == 2:
        rot = torch.cat((
            torch.cat((cosa, -sina), 2),
            torch.cat((sina, cosa), 2),
        ), 1)
    else:
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        rot = torch.cat((
            torch.cat((cosa, zeros, sina), 2),
            torch.cat((zeros, ones, zeros), 2),
            torch.cat((-sina, zeros, cosa), 2),
        ), 1)
    rot = rot.view(batch_size, *(1 for _ in size), dim, dim)
    episodic_coords = torch.matmul(rot, ego_coords.unsqueeze(-1)).squeeze(-1)

    pos = observations["gps"].view(batch_size, *(1 for _ in size), -1)
    if pos.size(-1) == dim:
        episodic_coords += pos
    elif dim == 2: # and pos.size(-1) == 3
        episodic_coords[..., 0] -= pos[..., 2]
        episodic_coords[..., 1] += pos[..., 0]
    else: # dim == 3 and pos.size(-1) == 2
        episodic_coords[..., 0] += pos[..., 1]
        episodic_coords[..., 2] -= pos[..., 0]
    return episodic_coords


class EgoMapCoordinatesGrid(nn.Module):
    r"""Module to compute episodic (x=forward, y=right) coordinates
    corresponding to every pixels of the ego map
    """

    def __init__(self,
        ego_map_size: int,
        ego_map_resolution: float,
        out_size: Optional[int] = None,
    ) -> None:
        r"""Constructor initializing internal coordinates grid from ego_map properties

        :param ego_map_size: Dimension of ego map in number of pixels
        :type ego_map_size: int
        :param ego_map_resolution: Resolution of ego map in meters per pixel
        :type ego_map_resolution: float
        :param out_size: Dimension of the output coordinates grid.
            (defaults to ego_map_size)
        :type out_size: int or None
        """

        super().__init__()
        bound = 0.5 * (ego_map_size  - 1) * ego_map_resolution
        if out_size is None:
            out_size = ego_map_size
        dim = torch.linspace(-bound, bound, out_size)
        grid = torch.stack(torch.meshgrid(-dim, dim, indexing='ij'), -1)
        self.register_buffer("grid", grid.view(1, out_size, out_size, 2, 1), False)

    def forward(self, observations: TensorDict) -> torch.FloatTensor:
        r"""Computes the coordinates grid centered and rotated
        around the current position and orientation of the agent
        for a batch of observations

        :param observations: Batch of observations as a dict containing (at least):
            - 'compass': Current orientation (yaw) of the agent
                relative to its starting orientation
                as a :math:`B\times 1` tensor,
                where :math:`B\in\mathbb{N}` is the batch size
            - 'gps': Current position (forward, right) of the agent
                relative to its starting position and orientation
                as a :math:`B\times 2` tensor
        :type observations: habitat_baselines.TensorDict
        :returns: A :math:`B\times D\times D\times 2` tensor containing
            the (x=right, z=back) coordinates of every pixels
            in a :math:`D\times D` grid centered around the agent
            and aligned with its orientation,
            where :math:`D\in\mathbb{N}` is the output size
            given at the initialization of the module
        :rtype: torch.FloatTensor
        """
        grid = self.grid.expand(observations["depth"].size(0), -1, -1, -1, -1)
        return _ego_to_episodic_coords(observations, grid)


class InverseCameraModel(nn.Module):
    r"""Module to project pixels from a depth image to world coordinates
    relative to the initial position and orientation of the agent
    """

    use_normalized_depth: bool
    depth_offset: float
    depth_range: float

    def __init__(self,
        width: int = 256,
        height: int = 256,
        hfov: float = 90,
        use_normalized_depth: bool = True,
        min_depth: float = 0.0,
        max_depth: float = 10.0,
        position: Tuple[float, float, float] = (0.0, 1.5, 0.0),
        orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        r"""Constructor initializing camera parameters from config

        :param width: Width of the depth image in pixels
            (defaults to 256)
        :type width: int
        :param height: Height of the depth image in pixels
            (defaults to 256)
        :type height: int
        :param hfov: Horizontal field of view of the depth camera in degrees
            (defaults to 90.0)
        :type hfov: float
        :param use_normalized_depth: Toggle whether or not depth values are normalized
            (defaults to True)
        :type use_normalized_depth: bool
        :param min_depth: Minimal depth value used for clipping and normalization
            (defaults to 0.0m)
        :type min_depth: float
        :param max_depth: Maximal depth value used for clipping and normalization
            (defaults to 10.0m)
        :type max_depth: float
        :param position: Position (x=right, y=up, z=back) of the depth camera
            relative to the base of the agent in meters
            (defaults to (0.0, 1.5, 0.0))
        :type position: tuple[float, float, float]
        :param orientation: Orientation (pitch, yaw, roll) of the depth camera
            relative to the agent in radians
            (defaults to (0.0, 0.0, 0.0))
        :type orientation: tuple[float, float, float]
        """

        super().__init__()
        tan_hfov = math.tan(math.radians(0.5 * hfov))
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-0.5, 0.5, width) * tan_hfov,
            torch.linspace(0.5, -0.5, height) * tan_hfov * height / width,
            indexing='xy'
        )
        z_grid = torch.full((height, width), -1.0)
        grid = torch.stack((x_grid, y_grid, z_grid), -1)
        self.register_buffer("grid", grid.view(1, height, width, 3))
        self.use_normalized_depth = use_normalized_depth
        self.depth_offset = min_depth
        self.depth_range = max_depth - min_depth
        self.register_buffer("sensor_position", torch.FloatTensor(position))
        cosp, cosy, cosr = (math.cos(a) for a in orientation)
        sinp, siny, sinr = (math.sin(a) for a in orientation)
        pitch = torch.FloatTensor([
            [1,    0,     0],
            [0, cosp, -sinp],
            [0, sinp,  cosp],
        ])
        yaw = torch.FloatTensor([
            [ cosy, 0, siny],
            [    0, 1,    0],
            [-siny, 0, cosy],
        ])
        roll = torch.FloatTensor([
            [cosr, -sinr, 0],
            [sinr,  cosr, 0],
            [   0,     0, 1],
        ])
        rotation = roll.mm(yaw.mm(pitch))
        self.register_buffer("sensor_rotation", rotation.view(1, 1, 1, 3, 3))

    def forward(self,
        observations: TensorDict,
        image_coords: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        r"""Computes a pointclound of every pixel in the depth image
        relative to the initial position and orientation of the agent
        at the beginning of the episode

        :param observations: Batch of observations as a dict containing (at least):
            - 'depth': Depth image captured from current position and orientation
                as a tensor of size :math:`B\times H\times W\times 1`,
                where :math:`B\in\mathbb{N}` is the batch size
                :math:`H\in\mathbb{N}` is the image height
                :math:`W\in\mathbb{N}` is the image width
            - 'compass': Current orientation (yaw) of the agent
                relative to its starting orientation
                as a :math:`B\times 1` tensor
            - 'gps': Current position (forward, right) of the agent
                relative to its starting position and orientation
                as a :math:`B\times 2` tensor
        :type observations: habitat_baselines.TensorDict
        :param image_coords: Batch of (i, j) pixels coordinates in the depth image
            to include in the point cloud as a :math:`B\times L\times 2` tensor
            (by default, include the whole depth image)
        :type image_coords: torch.LongTensor or None
        :returns: A :math:`B\times H\times W\times 3` tensor containing
            the (x=right, y=up, z=back) coordinates of every pixels in the depth image
        :rtype: torch.FloatTensor
        """

        depth = observations["depth"]
        if self.use_normalized_depth:
            depth = self.depth_range * depth + self.depth_offset
        pos_to_cam = (depth * self.grid).unsqueeze(-1)
        if image_coords is None:
            pos_to_agent = (
                torch.matmul(self.sensor_rotation, pos_to_cam).squeeze(-1)
                + self.sensor_position
            )
        else:
            batch_size, height, width, _ = depth.size()
            flat_i = image_coords[..., 0] * width + image_coords[..., 1]
            pos_to_cam = pos_to_cam.view(batch_size, height * width, 3, 1).gather(
                1, flat_i.view(batch_size, -1, 1, 1).expand(-1, -1, 3, 1)
            )
            pos_to_agent = (
                torch.matmul(self.sensor_rotation.squeeze(1), pos_to_cam).squeeze(-1)
                + self.sensor_position
            )
        return _ego_to_episodic_coords(observations, pos_to_agent)
