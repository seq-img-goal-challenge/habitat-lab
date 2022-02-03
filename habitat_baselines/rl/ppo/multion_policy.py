import math

import numpy as np
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, Net
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.models.multion import (
    MapCNN, Projection, RotateTensor, ToGrid, get_grid
)


@baseline_registry.register_policy
class MultiONPolicy(Policy):
    @classmethod
    def from_config(cls, config, observation_space, action_space):
        device = torch.device(
            f"cuda:{config.TORCH_GPU_ID}" if torch.cuda.is_available() else "cpu"
        )
        if config.RL.MULTION.agent_type != "non-oracle":
            net = MultiONOracleNet(
                config.RL.MULTION.agent_type,
                observation_space,
                config.RL.PPO.hidden_size,
                config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                device,
                config.RL.MULTION.object_category_embedding_size,
                config.RL.MULTION.previous_action_embedding_size,
                config.RL.MULTION.use_previous_action,
            )
        else:
            net = MultiONNonOracleNet(
                config.NUM_ENVIRONMENTS,
                observation_space,
                config.RL.PPO.hidden_size,
                config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                device,
                config.RL.MULTION.object_category_embedding_size,
                config.RL.MULTION.previous_action_embedding_size,
                config.RL.MULTION.use_previous_action,
                config.RL.MULTION.MAPS.egocentric_map_size,
                config.RL.MULTION.MAPS.global_map_size,
                config.RL.MULTION.MAPS.global_map_depth,
                config.RL.MULTION.MAPS.coordinate_min,
                config.RL.MULTION.MAPS.coordinate_max
            )
        return cls(net, action_space.n, config.RL.POLICY)

    def evaluate_actions(self, observations, rnn_hidden_states, prev_actions, masks, action):
        self.net.enable_map_update = False
        results = super().evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )
        self.net.enable_map_update = True
        return results


class MultiONOracleNet(Net):
    def __init__(self,
        agent_type,
        observation_space,
        hidden_size,
        goal_sensor_uuid,
        device,
        objet_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
    ):
        super().__init__()
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.visual_encoder = SimpleCNN(observation_space, 512)
        if agent_type == "oracle":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        elif agent_type == "no-map":
            self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        elif agent_type == "oracle-ego":
            self.map_encoder = MapCNN(50, 256, agent_type)
            self.object_embedding = nn.Embedding(10, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        if self.use_previous_action:
            self.state_encoder = build_rnn_state_encoder(
                hidden_size + object_category_embedding_size + previous_action_embedding_size,
                hidden_size,
            )
        else:
            self.state_encoder = build_rnn_state_encoder(
                hidden_size + object_category_embedding_size,
                hidden_size,
            )
        self.enable_map_update = True
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = observations[self.goal_sensor_uuid]
        x = [self.goal_embedding(
            (target_encoding).type(torch.LongTensor).to(self.device)
        ).squeeze(1)]
        bs = target_encoding.shape[0]
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        if self.agent_type != "no-map":
            global_map_embedding = []
            global_map = observations['ego_map']
            # TODO -> Split occupancy and goals into two chan
            if self.agent_type == "oracle":
                global_map_embedding.append(self.occupancy_embedding(
                    global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)
                ).view(bs, 50, 50 , -1))
            global_map_embedding.append(self.object_embedding(
                global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)
            ).view(bs, 50, 50, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x

        if self.use_previous_action:
            x = torch.cat(x + [self.action_embedding(prev_actions).squeeze(1)], dim=1)
        else:
            x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states  


class MultiONNonOracleNet(Net):
    def __init__(self,
        batch_size,
        observation_space,
        hidden_size,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        coordinate_min,
        coordinate_max
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space[self.goal_sensor_uuid].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth
        self.visual_encoder = SimpleCNN(observation_space, 0)
        self.map_encoder = MapCNN(51, 256, "non-oracle")        

        self.projection = Projection(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max
        )

        self.to_grid = ToGrid(global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        self.image_features_linear = nn.Linear(32 * 28 * 28, 512)

        self.flatten = nn.Flatten()

        if self.use_previous_action:
            self.state_encoder = build_rnn_state_encoder(
                512
                + 256
                + object_category_embedding_size
                + previous_action_embedding_size,
                self._hidden_size,
            )
        else:
            self.state_encoder = build_rnn_state_encoder(
                512
                + 256
                + object_category_embedding_size,
                self._hidden_size,
            )
        self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        self.full_global_map = torch.zeros(
            batch_size,
            global_map_size,
            global_map_size,
            global_map_depth,
            device=self.device,
        )
        self.enable_map_update = True
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        # TODO retrieve global_map
        target_encoding = observations[self.goal_sensor_uuid]
        goal_embed = self.goal_embedding(
            (target_encoding).type(torch.LongTensor).to(self.device)
        ).squeeze(1)
        
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
        projection = self.projection.forward(
            perception_embed, observations['depth'] * 10, -(observations["compass"])
        )
        perception_embed = self.image_features_linear(self.flatten(perception_embed))
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        bs = global_map.size(0)
        if self.enable_map_update:
            self.full_global_map[:bs, :, :, :] = (
                self.full_global_map[:bs, :, :, :] * masks.unsqueeze(1).unsqueeze(1)
            )
            if bs != 18:
                self.full_global_map[bs:, :, :, :] = self.full_global_map[bs:, :, :, :] * 0
            agent_view = torch.FloatTensor(
                bs, self.global_map_depth, self.global_map_size, self.global_map_size
            ).to(self.device).fill_(0)
            agent_view[:, :, 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):
                    self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):
                    self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            st_pose = torch.cat(
                [-(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                 -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                 observations['compass']], 
                 dim=1
            )
            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)
            self.full_global_map[:bs, :, :, :] = torch.max(
                self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1)
            )
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    (grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    torch.zeros_like(observations['compass'])
                    ],
                    dim=1
                )
            _, trans_mat_retrieval = get_grid(
                st_pose_retrieval, agent_view.size(), self.device
            )
            translated_retrieval = F.grid_sample(
                self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2), trans_mat_retrieval
            )
            translated_retrieval = translated_retrieval[:,:,
                self.global_map_size//2-math.floor(51/2):
                    self.global_map_size//2+math.ceil(51/2), 
                self.global_map_size//2-math.floor(51/2):
                    self.global_map_size//2+math.ceil(51/2)
            ]
            final_retrieval = self.rotate_tensor.forward(
                translated_retrieval, observations["compass"]
            )

            global_map_embed = self.map_encoder(final_retrieval.permute(0, 2, 3, 1))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((
                perception_embed, global_map_embed, goal_embed, action_embedding
            ), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        else: 
            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)  ##verify
            agent_view = torch.FloatTensor(
                bs, self.global_map_depth, 51, 51
            ).to(self.device).fill_(0)
            agent_view[:, :, 
                51//2 - math.floor(self.egocentric_map_size/2):
                    51//2 + math.ceil(self.egocentric_map_size/2), 
                51//2 - math.floor(self.egocentric_map_size/2):
                    51//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            
            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            global_map_embed = self.map_encoder(final_retrieval)

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((
                perception_embed, global_map_embed, goal_embed, action_embedding
            ), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        #TODO export global_map = final_retrieval.permute(0, 3, 1, 2)
        return x, rnn_hidden_states
