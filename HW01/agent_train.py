import random
import numpy as np
import os
import torch
from torch import nn

SEED = 42


def make_reproducible(seed, make_cuda_reproducible):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_cuda_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


make_reproducible(SEED, False)


def transform_state(state):
    state = (torch.tensor(state, requires_grad=False).float() + torch.tensor([-0.29426465, -0.0006308])) / torch.tensor([0.26634273, 0.00164501])
    #state = torch.tensor((state[0], state[1], torch.sign(state[1]), torch.sin(state[1]), torch.cos(state[1])), requires_grad=True)
    state = torch.tensor((state[0], state[1], torch.sign(state[1])),
                         requires_grad=True)
    return state


def transform_state_tensor(state):
    state = (state + torch.tensor([1.2, 0.0])) / torch.tensor([1.8, 0.07])
    return state


class Agent:
    def __init__(self):
        self.agent_filepath = "agent.pt"
        self.q_learning_net = nn.Linear(in_features=3, out_features=3, bias=True)
        self.q_learning_net.train()  # TODO: set to eval

    def act(self, state):
        return torch.argmax(self.forward(state)).item()

    def forward(self, state):
        transformed_state = transform_state(state)
        return self.q_learning_net(transformed_state)

    def reset(self):
        pass

