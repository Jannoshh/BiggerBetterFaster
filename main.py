import random
from dataclasses import dataclass
from itertools import count

import gymnasium as gym
import torch
from einops import einops

from configs.bbf import EnvironmentConfig, NetworkConfig, TrainingConfig
from models import DQN
from replay_buffer import ReplayBuffer
from utils import preprocess_state, epsilon_decay, TargetNetworkUpdater, shrink_and_perturb_parameters, \
    exponential_scheduler


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
        q_values = target_dqn(state)
        return q_values.argmax(dim=1).item()


# Initialize DQN and target DQN
dqn = DQN(state_shape, n_actions).float()
target_dqn = DQN(state_shape, n_actions).float()
target_dqn.load_state_dict(dqn.state_dict())
ema_updater = TargetNetworkUpdater(dqn, target_dqn, net_config.tau)

# Create the optimizer and the replay buffer
optimizer = torch.optim.AdamW(params=dqn.parameters(), lr=net_config.learning_rate, weight_decay=net_config.weight_decay)
replay_buffer = ReplayBuffer(capacity=net_config.buffer_size)
update_horizon_scheduler = exponential_scheduler(decay_period=train_config.cycle_steps,
                                                 initial_value=train_config.max_update_horizon,
                                                 final_value=train_config.min_update_horizon)
gamma_scheduler = exponential_scheduler(decay_period=train_config.cycle_steps,
                                        initial_value=train_config.min_gamma,
                                        final_value=train_config.max_gamma)


def optimize_model(update_horizon, gamma):
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
    target_q_values = rewards + net_config.gamma * next_q_values * mask

    loss = ((current_q_values - target_q_values) ** 2).mean()
    return loss


total_steps = 0
gradient_steps = 0
all_rewards = []

while total_steps < train_config.num_steps:
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

        for _ in range(train_config.replay_ratio):
            gradient_steps_since_reset = gradient_steps % train_config.reset_interval
            optimize_model(update_horizon=update_horizon_scheduler(gradient_steps_since_reset),
                           gamma=gamma_scheduler(gradient_steps_since_reset))
            gradient_steps += 1
            if gradient_steps % train_config.reset_interval == 0:
                shrink_and_perturb_parameters(dqn, train_config.alpha)


        ema_updater.soft_update()

        if done or total_steps >= train_config.num_steps:
            break

    all_rewards.append(episode_reward)
    print(f"Total Steps: {total_steps}, Last Episode Reward: {episode_reward}")

print("Training completed!")

