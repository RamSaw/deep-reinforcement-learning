import random
import numpy as np
import os
from .train import transform_state


class Agent:
    def __init__(self):
        if not os.path.exists(__file__[:-8] + "agent.npz"):
            self.weight, self.bias = np.random.normal(size=(5, 2)), np.random.normal(size=5)
        else:
            weights = np.load(__file__[:-8] + "agent.npz")
            self.weight, self.bias = weights['weight'], weights['bias']

    def act(self, state):
        return np.argmax(self.weight.dot(transform_state(state)) + self.bias)

    def reset(self):
        pass

