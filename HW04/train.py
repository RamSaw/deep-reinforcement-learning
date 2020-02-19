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
from torch.distributions import MultivariateNormal

from HW04.agent import Actor, transform_state

GAMMA = 0.99
CLIP = 0.2
EPISODES = 10000
ENTROPY_COEF = 1e-2
TRAJECTORY_SIZE = 3000  # TODO: change
POLICY_UPDATE_ITERATIONS = 80
EPOSIODE_LEN = 1000
BETAS = (0.9, 0.999)
LR = 0.0003
SEED = 5


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def reset(self):
        self.actions = []
        self.states = []
        self.logprobs = []
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
        action_mean, action_var = self.actor(state)
        cov_mat = torch.diag(action_var)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean, action_var = self.actor(state)

        action_var = action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self):
        self.policy = ActorCritic()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR, betas=BETAS)
        self.policy_old = ActorCritic()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.loss_func = nn.MSELoss()

    def select_action(self, state, memory):
        state = transform_state(state)
        return self.policy_old.act(state, memory).numpy()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        for _ in range(POLICY_UPDATE_ITERATIONS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss_func(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
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
            action = algo.select_action(state, memory)
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
