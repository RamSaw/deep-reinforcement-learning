import random
import numpy as np
import os

import torch
from torch import nn
from torch.distributions import MultivariateNormal


def transform_state(state):
    return torch.tensor(state).float()


ACTION_STD = 0.5


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(26, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
            nn.Tanh()
        )
        self.action_var = torch.full((6,), 0.5 * 0.5)

    def forward(self, x):
        mu = self.actor(x)
        return mu, self.action_var


class Agent:
    def __init__(self):
        self.actor = Actor()
        self.actor.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.actor.to(torch.device("cpu"))
        self.actor.eval()

    def act(self, state):
        state = transform_state(state)
        mu, sigma = self.actor(state)
        cov_mat = torch.diag(sigma)
        dist = MultivariateNormal(mu, cov_mat)
        out = dist.sample()
        return out.detach().cpu().numpy()

    def reset(self):
        pass

