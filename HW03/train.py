import time
from collections import deque

from gym import make
import numpy as np
import torch
import random

from torch import nn
from torch.distributions import Normal

from HW03.agent import transform_state, Agent

N_STEP = 64
GAMMA = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2C:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.actor = Agent.generate_model().to(DEVICE)  # Torch model
        self.critic = None # Torch model

    def update(self, transition):
        state, action, next_state, reward, done = transition

    def act(self, state):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        state = transform_state(state)
        out = self.actor(state)
        return np.array([Normal(out[0], out[1]).sample().item()])

    def save(self, i):
        torch.save(self.actor.state_dict(), f'agent_{i}.pkl')


if __name__ == "__main__":
    env = make("Pendulum-v0")
    a2c = A2C(state_dim=3, action_dim=1)
    episodes = 10000

    scores = []
    best_score = -10000.0
    best_score_10 = -10000.0
    total_steps = 0
    start = time.time()

    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque(maxlen=N_STEP)
        state_buffer = deque(maxlen=N_STEP)
        action_buffer = deque(maxlen=N_STEP)
        while not done:
            total_steps += 1
            action = a2c.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                a2c.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
            #env.render()
        scores.append(total_reward)
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                a2c.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if (i + 1) % 75 == 0:
            current_score = np.mean(scores)
            print(f'Current score: {current_score}')
            scores = []
            if current_score > best_score:
                best_score = current_score
                a2c.save(75)
                print(f'Best model saved with score: {best_score}')
            end = time.time()
            elapsed = end - start
            start = end
            print(f'Elapsed time: {elapsed}')
        elif (i + 1) % 25 == 0:
            current_score_10 = np.mean(scores[-10:])
            print(f'Intermediate score: {current_score_10}')
            if current_score_10 > best_score_10:
                best_score_10 = current_score_10
                a2c.save(10)
                print(f'Best 10 model saved with score: {best_score_10}')