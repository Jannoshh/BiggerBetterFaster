import random
from itertools import count

import gymnasium as gym
import torch
from einops import einops

from configs.bbf import EnvironmentConfig, NetworkConfig, TrainingConfig
from models import DQN, ImpalaCNN
from replay_buffer import ReplayBuffer
from utils import preprocess_state, TargetNetworkUpdater, \
    exponential_scheduler, linearly_decaying_epsilon


class BBFAgent(torch.nn.Module):
    def __init__(self, network=DQN):
        super(BBFAgent, self).__init__()

        # Store configurations
        self.env_config = EnvironmentConfig()
        self.net_config = NetworkConfig()
        self.train_config = TrainingConfig()

        # Environment setup
        self.env = gym.make(self.env_config.env_name)
        self.n_actions = self.env.action_space.n

        # Networks

        self.network = network()
        self.target_network = network()
        self.target_network.load_state_dict(self.network.state_dict())
        self.ema_updater = TargetNetworkUpdater(self.network, self.target_network, self.net_config.tau)

        # Optimizer and Replay Buffer
        self.optimizer = torch.optim.AdamW(params=self.network.parameters(),
                                           lr=self.net_config.learning_rate,
                                           weight_decay=self.net_config.weight_decay)
        self.replay_buffer = ReplayBuffer(capacity=self.net_config.buffer_size)

        # Schedulers
        self.update_horizon_scheduler = exponential_scheduler(
            decay_period=self.train_config.cycle_steps,
            initial_value=self.train_config.max_update_horizon,
            final_value=self.train_config.min_update_horizon
        )
        self.gamma_scheduler = exponential_scheduler(
            decay_period=self.train_config.cycle_steps,
            initial_value=self.train_config.min_gamma,
            final_value=self.train_config.max_gamma
        )

    def shrink_and_perturb_parameters(self):
        for name, param in self.model.named_parameters():
            if 'encoder' in name:  # Assuming 'encoder' is part of the name for encoder layers
                phi = torch.randn_like(param).normal_(0, 0.01)  # Assuming a normal initializer with mean 0 and std 0.01
                param.data = self.train_config.alpha * param.data + (1 - self.train_config.alpha) * phi
            elif 'final' in name:
                phi = torch.randn_like(param).normal_(0, 0.01)  # Assuming a normal initializer with mean 0 and std 0.01
                param.data = phi

    def select_epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state = einops.rearrange(state, 'c h w -> 1 c h w')
            q_values = self.target_network(state)
            return q_values.argmax(dim=1).item()

    def optimize_model(self, update_horizon, gamma):
        if len(self.replay_buffer) > self.net_config.batch_size:
            batch = self.replay_buffer.sample(self.net_config.batch_size)
            loss = self.compute_loss(batch, update_horizon, gamma)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, batch, update_horizon, gamma):
        states, actions, rewards, next_states, dones = batch
        mask = 1 - dones
        current_q_values = self.network(states).gather(1, einops.rearrange(actions, 'b -> b 1'))
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * mask
        loss = ((current_q_values - target_q_values) ** 2).mean()
        return loss

    def train(self):
        total_steps = 0
        gradient_steps = 0
        all_rewards = []

        while total_steps < self.train_config.num_steps:
            state, _ = self.env.reset()
            state = preprocess_state(state)
            episode_reward = 0

            for episode_step in count():
                epsilon = linearly_decaying_epsilon(
                    decay_period=self.train_config.epsilon_decay_period,
                    step=total_steps,
                    warmup_steps=0,
                    epsilon=self.train_config.epsilon_train
                )
                action = self.select_epsilon_greedy_action(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = preprocess_state(next_state)
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                total_steps += 1

                for _ in range(self.train_config.replay_ratio):
                    gradient_steps_since_reset = gradient_steps % self.train_config.reset_interval
                    self.optimize_model(
                        update_horizon=self.update_horizon_scheduler(gradient_steps_since_reset),
                        gamma=self.gamma_scheduler(gradient_steps_since_reset)
                    )
                    gradient_steps += 1
                    if gradient_steps % self.train_config.reset_interval == 0:
                        self.shrink_and_perturb_parameters()

                self.ema_updater.soft_update()

                if done or total_steps >= self.train_config.num_steps:
                    break

            all_rewards.append(episode_reward)
            print(f"Total Steps: {total_steps}, Last Episode Reward: {episode_reward}")

        print("Training completed!")


# Instantiating the agent
agent = BBFAgent()
agent.train()

