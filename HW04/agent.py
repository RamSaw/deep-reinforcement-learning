import random
import numpy as np
import os

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal


def transform_state(state):
    return torch.tensor(state).float()


ACTION_STD = 0.5


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(26, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.sigma = nn.Parameter(torch.zeros(6))

    def forward(self, x):
        mu = self.actor(x)
        return mu, self.sigma.exp()


class Agent:
    def __init__(self):
        self.actor = Actor()
        self.actor.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.actor.to(torch.device("cpu"))
        self.actor.eval()

    def act(self, state):
        state = transform_state(state)
        mu, sigma = self.actor(state)
        dist = Normal(mu, sigma)
        out = dist.sample()
        return out.detach().cpu().numpy()

    def reset(self):
        pass

