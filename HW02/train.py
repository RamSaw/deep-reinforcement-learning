import random
from collections import deque

import numpy as np
import torch
from gym import make

from HW02.agent import Agent, act

N_STEP = 1
GAMMA = 0.9


class DQN:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.model = Agent.generate_model()

    def update(self, transition):
        state, action, next_state, reward, done = transition

    def act(self, state, target=False):
        return act(self.model, state)

    def save(self, path):
        torch.save(self.model.state_dict(), "agent.pkl")


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=8, action_dim=4)
    eps = 0.1
    episodes = 1000
    scores = []
    best_score = -1000.0

    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                dqn.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
            #env.render()
        scores.append(total_reward)
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                dqn.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if (i + 1) % 50 == 0:
            current_score = np.mean(scores)
            print(f'Current score: {current_score}')
            scores = []
            if current_score > best_score:
                best_score = current_score
                dqn.save(__file__[:-8])
