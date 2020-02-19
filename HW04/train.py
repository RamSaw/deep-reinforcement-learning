import os
import time
from collections import deque

import pybullet_envs
import sklearn
from gym import make
import numpy as np
import torch
import random

from torch import nn
from torch.distributions import MultivariateNormal, Normal
from HW04.agent import Actor, transform_state, Agent

GAMMA = 0.99
CLIP = 0.2
EPISODES = 10000
ENTROPY_COEF = 1e-2
TRAJECTORY_SIZE = 3000
POLICY_UPDATE_ITERATIONS = 80
VALUE_FUNCTION_LOSS_COEF = 0.5
EPOSIODE_LEN = 1000
BETAS = (0.9, 0.999)
LR = 0.0003
SEED = 5


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def reset(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.actor = Actor()
        self.critic = nn.Sequential(
            nn.Linear(26, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def act(self, state, memory):
        mu, sigma = self.actor(state)
        dist = Normal(mu, sigma)
        sampled_action = dist.sample()
        action_log_prob = dist.log_prob(sampled_action)

        memory.states.append(state)
        memory.actions.append(sampled_action)
        memory.log_probs.append(action_log_prob)

        return sampled_action.detach()

    def evaluate(self, state, action):
        mu, sigma = self.actor(state)
        sigma = sigma.expand_as(mu)
        dist = Normal(mu, sigma)
        return dist.log_prob(action), dist.entropy(), torch.squeeze(self.critic(state), 1)


def normalize(tensor, eps=1e-10):
    return (tensor - tensor.mean()) / (tensor.std() + eps)


class PPO:
    def __init__(self):
        self.policy = ActorCritic()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR, betas=BETAS)
        self.policy_old = ActorCritic()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.loss_func = nn.MSELoss()

    def act(self, state, memory):
        state = transform_state(state)
        return self.policy_old.act(state, memory).numpy()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.append(discounted_reward)

        rewards = normalize(torch.tensor(list(reversed(rewards))))

        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_log_probs = torch.stack(memory.log_probs).detach()

        for _ in range(POLICY_UPDATE_ITERATIONS):
            log_probs, entropy, state_values = self.policy.evaluate(old_states, old_actions)

            entropy_loss = -entropy * ENTROPY_COEF
            value_loss = VALUE_FUNCTION_LOSS_COEF * self.loss_func(state_values, rewards)
            adv = rewards - state_values.detach()
            adv = adv.unsqueeze(dim=1)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            policy_loss = -(torch.min(ratios * adv, torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * adv))
            loss = (policy_loss + value_loss + entropy_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, i):
        torch.save(self.policy.actor.state_dict(), f'agent_{i}.pkl')


if __name__ == "__main__":
    env = make("HalfCheetahBulletEnv-v0")

    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.seed(SEED)

    memory = Memory()
    algo = PPO()

    scores = []
    best_score = -10000.0
    best_score_intermediate = -10000.0
    intermediate_i = 10
    total_steps = 0
    start = time.time()

    # env.render()
    for i in range(EPISODES):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            if steps == EPOSIODE_LEN:
                break
            action = algo.act(state, memory)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if len(memory.rewards) == TRAJECTORY_SIZE:
                algo.update(memory)
                memory.reset()

        scores.append(total_reward)

        if (i + 1) % 50 == 0:
            current_score = np.mean(scores)
            print(f'Current score: {current_score}')
            scores = []
            if current_score > best_score:
                best_score = current_score
                algo.save(50)
                print(f'Best model saved with score: {best_score}')
            end = time.time()
            elapsed = end - start
            start = end
            print(f'Elapsed time: {elapsed}')
        elif (i + 1) % intermediate_i == 0:
            current_score_intermediate = np.mean(scores[-intermediate_i:])
            print(f'Intermediate score: {current_score_intermediate}')
            if current_score_intermediate > best_score_intermediate:
                best_score_intermediate = current_score_intermediate
                algo.save(intermediate_i)
                print(f'Best {intermediate_i} model saved with score: {best_score_intermediate}')
