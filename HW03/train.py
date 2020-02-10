import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from gym import make
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam

from HW03.agent import transform_state, Agent

N_STEP = 1
GAMMA = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPDATE_LENGTH = 10
ENTROPY = 0.01
TARGET_UPDATE = 800
EPOSIODE_LEN = 200


class A2C:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.actor = Agent.generate_model().to(DEVICE)  # Torch model
        self.critic = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 1)).to(DEVICE)  # Torch model
        self.actor_optimizer = Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=0.001)
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        self.critic_values = []
        self.target_values = []
        self.update_steps = 0
        self.distribution = None
        self.target = deepcopy(self.critic)

    def optimizer_step(self, log_probs, entropies, returns, critic_values):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        adv = (returns - critic_values).detach()
        policy_loss = -(log_probs * adv).mean()
        entropy_loss = -entropies.mean() * ENTROPY
        value_loss = ((critic_values - returns) ** 2 / 2).mean()
        total_loss = value_loss + entropy_loss + policy_loss
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def update(self, transition):
        self.update_steps += 1
        if self.update_steps % TARGET_UPDATE == 0:
            self.target = deepcopy(self.critic)

        state, action, next_state, reward, done = transition
        action = torch.tensor(action)
        self.states.append(transform_state(state))
        self.actions.append(action)
        self.next_states.append(transform_state(next_state))
        self.rewards.append((reward + 8.1) / 8.1)
        self.log_probs.append(self.distribution.log_prob(action))
        self.entropies.append(self.distribution.entropy())
        self.critic_values.append(self.critic(transform_state(state)))
        self.target_values.append(self.target(transform_state(state)))
        if done or len(self.states) == UPDATE_LENGTH:
            next_target = torch.zeros(1) if done else self.target(self.next_states[-1])
            if len(self.target_values) > 1:
                returns = torch.tensor(self.rewards).view(-1, 1) + \
                          self.gamma * torch.cat((torch.cat(self.target_values[1:]), next_target)).view(-1, 1).detach()
            else:
                self.states = []
                self.actions = []
                self.next_states = []
                self.rewards = []
                self.log_probs = []
                self.entropies = []
                self.critic_values = []
                self.target_values = []
                return

            self.optimizer_step(
                torch.cat(self.log_probs).view(-1, 1),
                torch.cat(self.entropies).view(-1, 1),
                returns,
                torch.cat(self.critic_values).view(-1, 1)
            )

            self.states = []
            self.actions = []
            self.next_states = []
            self.rewards = []
            self.log_probs = []
            self.entropies = []
            self.critic_values = []
            self.target_values = []

    def act(self, state):
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        state = transform_state(state)
        out = self.actor(state)
        self.distribution = Normal(out[0], out[1])
        return np.array([self.distribution.sample().item()])

    def save(self, i):
        torch.save(self.actor.state_dict(), f'agent_{i}.pkl')


if __name__ == "__main__":
    env = make("Pendulum-v0")
    a2c = A2C(state_dim=3, action_dim=1)
    episodes = 10000

    scores = []
    best_score = -10000.0
    best_score_25 = -10000.0
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
            if steps == EPOSIODE_LEN:
                break
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
        #elif (i + 1) % 25 == 0:
        #    current_score_25 = np.mean(scores[-25:])
        #    print(f'Intermediate score: {current_score_25}')
        #    if current_score_25 > best_score_25:
        #        best_score_25 = current_score_25
        #        a2c.save(25)
        #        print(f'Best 25 model saved with score: {best_score_25}')