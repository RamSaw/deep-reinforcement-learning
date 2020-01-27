import random
from collections import deque

import numpy as np
import torch
from gym import make
from torch import optim, nn

N_STEP = 3
GAMMA = 0.96


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(result)


class AQL:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        from HW01.agent import Agent
        self.agent = Agent()
        self.cur = 0
        self.optimizer = optim.Adam(self.agent.q_learning_net.parameters(), lr=0.00001)
        self.loss_function = nn.MSELoss()

    def update(self, transition):
        self.optimizer.zero_grad()

        state, action, next_state, reward, done = transition
        target = reward + self.gamma * self.agent.act(next_state)
        output = self.agent.q_learning_net(torch.tensor(transform_state(state), requires_grad=True).float())[action]
        loss = self.loss_function(output.float(), target)
        loss.backward()
        self.optimizer.step()

    def act(self, state, target=False):
        return self.agent.act(state).item()

    def save(self, path):
        torch.save(self.agent.q_learning_net.state_dict(), self.agent.agent_filepath)


if __name__ == "__main__":
    env = make("MountainCar-v0")
    aql = AQL(state_dim=2, action_dim=3)
    eps = 0.1
    episodes = 2000

    scores = []
    best_score = -201.0
    for i in range(episodes):
        state = transform_state(env.reset())
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
                action = aql.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
            #env.render()
        scores.append(-steps)
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))

        if (i + 1) % 20 == 0:
            current_score = np.mean(scores)
            print(f'Current score: {current_score}')
            scores = []
            if current_score > best_score:
                best_score = current_score
                aql.save(__file__[:-8])
