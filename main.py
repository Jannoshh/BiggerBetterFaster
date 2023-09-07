import random
from dataclasses import dataclass
from itertools import count

import gymnasium as gym
import numpy as np
import torch
from einops import einops

from models import DQN
from replay_buffer import ReplayBuffer
from utils import preprocess_state, epsilon_decay


@dataclass
class EnvironmentConfig:
    env_name: str = 'ALE/BattleZone-v5'

@dataclass
class NetworkConfig:
    buffer_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4

@dataclass
class TrainingConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000
    num_episodes: int = 1000
    target_update_interval: int = 1000
    epsilon_decay_last_frame: int = 100000

# Initialize configurations
env_config = EnvironmentConfig()
net_config = NetworkConfig()
train_config = TrainingConfig()

# Use configurations in the code:
env = gym.make(env_config.env_name)
n_actions = env.action_space.n

sample_state, info = env.reset()
sample_state_preprocessed = preprocess_state(sample_state)
state_shape = sample_state_preprocessed.shape



def select_epsilon_greedy_action(model, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        state = einops.rearrange(torch.FloatTensor(state), 'c h w -> 1 c h w')
        q_values = model(state)
        return q_values.argmax(dim=1).item()


# Initialize DQN and target DQN
dqn = DQN(state_shape, n_actions).float()
target_dqn = DQN(state_shape, n_actions).float()
optimizer = torch.optim.Adam(dqn.parameters(), lr=net_config.learning_rate)
replay_buffer = ReplayBuffer(net_config.buffer_size)

# Create the optimizer and the replay buffer
optimizer = torch.optim.Adam(dqn.parameters(), lr=net_config.learning_rate)
replay_buffer = ReplayBuffer(net_config.buffer_size)

def optimize_model():
    if len(replay_buffer) > net_config.batch_size:
        batch = replay_buffer.sample(net_config.batch_size)
        loss = compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def compute_loss(batch):
    states, actions, rewards, next_states, dones = batch

    mask = 1 - dones
    current_q_values = dqn(states).gather(1, einops.rearrange(actions, 'b -> b 1'))
    next_q_values = target_dqn(next_states).max(1)[0]
    target_q_values = rewards + net_config.gamma * next_q_values * mask.squeeze()

    loss = ((current_q_values - target_q_values) ** 2).mean()
    return loss

total_steps = 0
all_rewards = []
episode_reward = 0

for episode in range(train_config.num_episodes):
    state, info = env.reset()  # Reset the environment
    state = preprocess_state(state)
    episode_reward = 0

    for episode_step in count():
        epsilon = epsilon_decay(total_steps, train_config.epsilon_start, train_config.epsilon_end, train_config.epsilon_decay_last_frame)
        action = select_epsilon_greedy_action(dqn, state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = preprocess_state(next_state)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        total_steps += 1

        optimize_model()

        if total_steps % train_config.target_update_interval == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if done:
            break

    all_rewards.append(episode_reward)
    print(f"Episode: {episode}, Reward: {episode_reward}")

print("Training completed!")

