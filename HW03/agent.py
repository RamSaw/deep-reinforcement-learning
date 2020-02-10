import random
import numpy as np
import os

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

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
    def __init__(self, hidden=400, in_dim=3, out_dim=1):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)
        self.mu = nn.Linear(hidden, out_dim)
        self.sigma = nn.Parameter(torch.full((1,), np.log(0.6)), requires_grad=True)

    def forward(self, x):
        z2 = F.relu(self.fc(x))
        mu = 2 * F.tanh(self.mu(z2))
        sigma = self.sigma.expand_as(mu).exp()
        return mu, sigma


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
