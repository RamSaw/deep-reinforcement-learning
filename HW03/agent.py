import os
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

SEED = 423  # 627, 8, 11


def make_reproducible(seed, make_cuda_reproducible):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_cuda_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


make_reproducible(SEED, make_cuda_reproducible=False)


def transform_state(state):
    return torch.tensor(state).float()


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_model = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.sigma = nn.Parameter(torch.log(torch.tensor(0.6)), requires_grad=True)

    def forward(self, x):
        mu = 2 * self.mu_model(x)
        return mu, self.sigma


class Agent:
    def __init__(self):
        self.actor = Agent.generate_model()
        self.actor.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.actor.to(torch.device("cpu"))
        self.actor.eval()

    @staticmethod
    def generate_model():
        return Actor()

    def act(self, state):
        state = transform_state(state)
        out = self.actor(state)
        return np.array([Normal(out[0], out[1]).sample().item()])

    def reset(self):
        pass
