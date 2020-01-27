import random
import numpy as np
import os
import torch
from torch import nn

from .train import transform_state

SEED = 41


def make_reproducible(seed, make_cuda_reproducible):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_cuda_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


make_reproducible(SEED, False)


class Agent:
    def __init__(self):
        self.agent_filepath = __file__[:-8] + "agent.pt"
        self.q_learning_net = nn.Sequential(
            nn.Linear(in_features=2, out_features=3, bias=True)
        )
        if not os.path.exists(self.agent_filepath):
            torch.save(self.q_learning_net.state_dict(), self.agent_filepath)
        self.q_learning_net.load_state_dict(torch.load(self.agent_filepath))
        self.q_learning_net.train()  # TODO: set to eval

    def act(self, state):
        transformed_state = torch.tensor(transform_state(state)).float()
        return torch.argmax(self.q_learning_net(transformed_state)).item()

    def reset(self):
        pass

