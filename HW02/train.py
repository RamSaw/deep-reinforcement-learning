import math
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from gym import make
from torch import optim

from HW02.agent import Agent, transform_state

N_STEP = 1
GAMMA = 0.9
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_CAPACITY = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.model = Agent.generate_model().to(DEVICE)
        self.target = Agent.generate_model().to(DEVICE)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(MEMORY_CAPACITY)

    def optimizer_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update(self, transition):
        state, action, next_state, reward, done = transition
        self.memory.push(torch.tensor(state, device=DEVICE),
                         torch.tensor([action], device=DEVICE),
                         torch.tensor(next_state, device=DEVICE) if not done else None,
                         torch.tensor([reward], device=DEVICE))
        self.optimizer_step()

    def act(self, state, target=False):
        state = transform_state(state).to(DEVICE)
        return torch.argmax(self.model(state)).item()

    def save(self, path):
        torch.save(self.model.state_dict(), "agent.pkl")


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=8, action_dim=4)
    episodes = 1000
    scores = []
    best_score = -1000.0
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
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY)
            total_steps += 1
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

        if (i + 1) % TARGET_UPDATE == 0:
            dqn.target.load_state_dict(dqn.model.state_dict())

        if (i + 1) % 50 == 0:
            current_score = np.mean(scores)
            print(f'Current score: {current_score}')
            scores = []
            if current_score > best_score:
                best_score = current_score
                dqn.save(None)
                print(f'Best model saved with best score: {best_score}')
            end = time.time()
            elapsed = end - start
            start = end
            print(f'Elapsed time: {elapsed}')
        elif (i + 1) % 10 == 0:
            print(f'Intermediate score: {np.mean(scores)}')
