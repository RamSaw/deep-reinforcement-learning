import time
from collections import deque

import pybullet_envs
from gym import make
import numpy as np
import torch
import random

from HW04.agent import Agent, transform_state

N_STEP = 1
GAMMA = 0.9
CLIP = 0.1
ENTROPY_COEF = 1e-2
TRAJECTORY_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.actor = Agent.generate_model().to(DEVICE)  # Torch model
        self.critic = None  # Torch model

    def update(self, trajectory):
        state, action, rollouted_reward = zip(*trajectory)

    def get_value(self, state):
        # Should return expected value of the state
        return 0

    def act(self, state):
        state = transform_state(state)
        out = self.actor(state)
        return out.detach().cpu().numpy()

    def save(self, i):
        torch.save(self.actor.state_dict(), f'agent_{i}.pkl')


if __name__ == "__main__":
    env = make("HalfCheetahBulletEnv-v0")
    algo = PPO(state_dim=26, action_dim=6)
    episodes = 10000

    scores = []
    best_score = -10000.0
    best_score_intermediate = -10000.0
    intermediate_i = 10
    total_steps = 0
    start = time.time()

    reward_buffer = deque()
    state_buffer = deque()
    action_buffer = deque()
    done_buffer = deque()
    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = algo.act(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            done_buffer.append(done)
            if len(action_buffer) == TRAJECTORY_SIZE:
                rollouted_reward = [algo.get_value(state) if not done else 0]
                for r, d in zip(reward_buffer, done_buffer):
                    rollouted_reward.append(
                        r + GAMMA * d * rollouted_reward[-1])  # TODO: * rb[-1]? rb = list(reward_buffer)
                rollouted_reward = list(reversed(rollouted_reward))
                trajectory = []
                for k in range(0, len(state_buffer)):
                    trajectory.append((state_buffer[k], action_buffer[k], rollouted_reward[k]))
                algo.update(trajectory)
                action_buffer.clear()
                reward_buffer.clear()
                state_buffer.clear()
                done_buffer.clear()
            # env.render()
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
