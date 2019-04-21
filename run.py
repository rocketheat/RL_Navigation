from unityagents import UnityEnvironment
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import namedtuple, deque
from time import sleep

# from model import QNetwork
from train import Agent

env = UnityEnvironment(file_name="./Banana.app")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size
# print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
# print('States look like:', state)
state_size = len(state)
# print('States have length:', state_size)

# Training hyperparameters
N_EPISODES=10
MAX_T=1000
EPS_START=1.0
EPS_END=0.01
EPS_DECAY=0.995

# Neural Network hyperparameters
FC1_UNITS=64
FC2_UNITS=64

# Deep Q-Learning hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# Average Reward Gain Target per 100 episodes
Target_Reward = 13.0

agent = Agent(
    state_size=state_size,
    action_size=action_size,
    seed=0,
    fc1_units=FC1_UNITS,
    fc2_units=FC2_UNITS,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    tau=TAU,
    lr=LR,
    update_every=UPDATE_EVERY,
    filename='./model.pth')

def dqn(n_episodes=N_EPISODES, max_t=MAX_T, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,\
        train=False):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0] # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name] # send the action to the environment
            next_state = env_info.vector_observations[0] # get the next state
            reward = env_info.rewards[0] # get the reward
            done = env_info.local_done[0] # see if episode has finished

            # if train:
            #     agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            sleep(0.01) # Time in seconds.

            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=Target_Reward:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'model.pth')
            break
    return scores

scores = dqn()

env.close()
