import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["MAGNUM_LOG"] = "quiet"
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import habitat
from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


class DummyPolicy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.conv_obs1 = nn.Conv2d(4, 32, 9)
        self.conv_obs2 = nn.Conv2d(32, 64, 9)
        self.conv_obs3 = nn.Conv2d(64, 16, 5)

        self.conv_goal1 = nn.Conv2d(3, 32, 9)
        self.conv_goal2 = nn.Conv2d(32, 64, 9)
        self.conv_goal3 = nn.Conv2d(64, 16, 5)

        peek_obs = np.concatenate((obs_space["rgb"].sample().astype(np.float32) / 255,
                                   obs_space["depth"].sample()), 2)
        peek_obs = torch.from_numpy(peek_obs).permute(2, 0, 1).unsqueeze(0)
        peek_obs_out = self.forward_conv_obs(peek_obs)
        peek_goal = obs_space["objectgoal_appearance"].sample().astype(np.float32) / 255
        peek_goal = torch.from_numpy(peek_goal).permute(0, 3, 1, 2)
        peek_goal_out = self.forward_conv_goal(peek_goal).sum(dim=0, keepdim=True)

        self.gru = nn.GRUCell(peek_obs_out.numel() + peek_goal_out.numel(), 1024)
        self.hidden = None
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, action_space.n)

    def reset(self):
        self.hidden = torch.zeros((1, 1024))

    def forward_conv_obs(self, rgbd):
        h1 = F.max_pool2d(self.conv_obs1(rgbd), 3)
        h2 = F.max_pool2d(self.conv_obs2(h1), 3)
        h3 = F.max_pool2d(self.conv_obs3(h2), 3)
        return h3

    def forward_conv_goal(self, goal):
        h1 = F.max_pool2d(self.conv_goal1(goal), 3)
        h2 = F.max_pool2d(self.conv_goal2(h1), 3)
        h3 = F.max_pool2d(self.conv_goal3(h2), 3)
        return h3

    def forward(self, obs):
        if self.hidden is None:
            raise RuntimeError("Call reset() before using this policy module.")
        rgbd = np.concatenate((obs["rgb"].astype(np.float32) / 255, obs["depth"]), 2)
        rgbd = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0)
        rgbd_out = self.forward_conv_obs(rgbd)

        goal = obs["objectgoal_appearance"].astype(np.float32) / 255
        goal = torch.from_numpy(goal).permute(0, 3, 1, 2)
        goal_out = self.forward_conv_goal(goal).sum(dim=0, keepdim=True)

        x = torch.cat((rgbd_out.view(1, -1), goal_out.view(1, -1)), 1)
        self.hidden = self.gru(x, self.hidden)
        h1 = F.relu(self.fc1(self.hidden)) 
        out = F.softmax(self.fc2(h1), 1) 
        return out


class DummyAgent(Agent):
    def __init__(self, env, epsilon=0.5):
        self._env = env
        self._policy = DummyPolicy(env.observation_space, env.action_space)
        self._follower = ShortestPathFollower(env.sim, env.task.success_dist, False, False)
        self._epsilon = epsilon

    def reset(self):
        self._policy.reset()

    def act(self, observations):
        if torch.rand((1,)) < self._epsilon:
            p = self._policy(observations)
            return Categorical(p).sample().item()
        else:
            episode = self._env.current_episode
            step = episode.steps[episode._current_step_index]
            goal_pos = np.array(step.goals[0].position)
            return self._follower.get_next_action(goal_pos)


if __name__ == "__main__":
    habitat.logger.setLevel(logging.ERROR)
    bench = Benchmark("configs/tasks/sequential_objectnav.yaml")
    agent = DummyAgent(bench._env, 0)
    m = bench.evaluate(agent)
    print(m)
