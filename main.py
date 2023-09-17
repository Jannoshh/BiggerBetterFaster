import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, TransformReward, \
    TransformObservation

import wandb
from einops import einops

from configs.bbf import EnvironmentConfig, NetworkConfig, TrainingConfig
from models import DQN
from replay_buffer import ReplayBuffer
from utils import TargetNetworkUpdater, \
    exponential_scheduler, linearly_decaying_epsilon


class BBFAgent(torch.nn.Module):
    def __init__(self, network=DQN):
        super(BBFAgent, self).__init__()

        # Store configurations
        self.env_config = EnvironmentConfig()
        self.network_config = NetworkConfig()
        self.train_config = TrainingConfig()

        # Environment setup
        env = gym.make(self.env_config.env_name)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=(84, 84))
        env = TransformObservation(env, lambda obs: obs / 255)
        env = FrameStack(env, num_stack=self.train_config.frames_stack)
        # env = TransformObservation(env, lambda obs: einops.rearrange(np.array(obs), "f h w c -> (c f) h w"))
        self.env = env
        self.n_actions = self.env.action_space.n
        sample_state, _ = self.env.reset()

        # Networks
        self.network = network(sample_state.shape, self.n_actions)
        self.target_network = network(sample_state.shape, self.n_actions)
        self.target_network.load_state_dict(self.network.state_dict())
        self.ema_updater = TargetNetworkUpdater(self.network, self.target_network, self.network_config.tau)

        self.optimizer = torch.optim.AdamW(params=self.network.parameters(),
                                           lr=self.network_config.learning_rate,
                                           weight_decay=self.network_config.weight_decay)
        self.replay_buffer = ReplayBuffer(capacity=self.network_config.buffer_size)

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

        self.gradient_steps = 0

    def shrink_and_perturb_parameters(self):
        return
        # for name, param in self.model.named_parameters():
        #     if 'encoder' in name:  # Assuming 'encoder' is part of the name for encoder layers
        #         phi = torch.randn_like(param).normal_(0, 0.01)  # Assuming a normal initializer with mean 0 and std 0.01
        #         param.data = self.train_config.alpha * param.data + (1 - self.train_config.alpha) * phi
        #     elif 'final' in name:
        #         phi = torch.randn_like(param).normal_(0, 0.01)  # Assuming a normal initializer with mean 0 and std 0.01
        #         param.data = phi

    def select_epsilon_greedy_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float32)
            state = einops.rearrange(state, 'c h w -> 1 c h w')
            q_values = self.target_network(state)
            return q_values.argmax(dim=1).item()

    def optimize_model(self, update_horizon, gamma):
        batch = self.replay_buffer.sample(self.network_config.batch_size)
        loss = self.compute_loss(batch, update_horizon, gamma)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def compute_loss(self, batch, update_horizon, gamma):
        states, actions, rewards, next_states, dones = batch
        mask = 1 - dones
        current_q_values = self.network(states).gather(1, einops.rearrange(actions, 'b -> b 1'))
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = torch.sign(rewards) + gamma * next_q_values * mask
        loss = ((current_q_values - target_q_values) ** 2).mean()
        return loss

    def train(self, project_name="bbf", run_name=None, disable_wandb=False):
        wandb.init(
            project=project_name,
            mode="disabled" if disable_wandb else "online",
            name=run_name,
        )

        all_rewards = []

        state, _ = self.env.reset()
        episode_reward = 0

        for step in range(self.train_config.num_steps):
            epsilon = linearly_decaying_epsilon(
                decay_period=self.train_config.epsilon_decay_period,
                step=step,
                warmup_steps=self.network_config.min_replay_history,
                epsilon=self.train_config.epsilon_train
            )
            action = self.select_epsilon_greedy_action(state, epsilon)
            accumulated_reward = 0
            for _ in range(self.train_config.action_repeat):
                next_state, reward, done, _, _ = self.env.step(action)
                accumulated_reward += reward
                if done:
                    break

            self.replay_buffer.push(state, action, accumulated_reward, next_state, done)

            state = next_state
            episode_reward += accumulated_reward

            if len(self.replay_buffer) > self.network_config.min_replay_history:
                for _ in range(self.train_config.replay_ratio):
                    self.train_step()

            self.ema_updater.soft_update()

            if done:
                all_rewards.append(episode_reward)
                print(f"Total Steps: {step}, Last Episode Reward: {episode_reward}")
                wandb.log({"episode_reward": episode_reward, "total_steps": step})
                state, _ = self.env.reset()
                episode_reward = 0

        wandb.finish()
        print("Training completed!")

    def train_step(self):
        gradient_steps_since_reset = self.gradient_steps % self.train_config.reset_interval
        self.optimize_model(
            update_horizon=self.update_horizon_scheduler(gradient_steps_since_reset),
            gamma=self.gamma_scheduler(gradient_steps_since_reset)
        )
        self.gradient_steps += 1
        if self.gradient_steps % self.train_config.reset_interval == 0:
            self.shrink_and_perturb_parameters()


agent = BBFAgent()
agent.train(disable_wandb=True)
