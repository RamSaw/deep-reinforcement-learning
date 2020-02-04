import os
import random

import numpy as np
import torch
from torch import nn

SEED = 11  # 627, 8, 11


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
    return torch.tensor(state)


class Agent:
    def __init__(self):
        self.model = Agent.generate_model()
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.to(torch.device("cpu"))
        self.model.eval()

    @staticmethod
    def generate_model():
        linears = [nn.Linear(8, 128), nn.Linear(128, 64), nn.Linear(64, 4)]
        #for l in linears:
        #    nn.init.uniform_(l.weight)
        #    nn.init.uniform_(l.bias)
        return nn.Sequential(
                linears[0],
                nn.ReLU(),
                linears[1],
                nn.ReLU(),
                linears[2])

    def act(self, state):
        with torch.no_grad():
            state = transform_state(state)
            return torch.argmax(self.model(state)).item()

    def reset(self):
        pass

