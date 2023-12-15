import random

import gymnasium as gym
import numpy as np
import torch
from einops import einops
from jaxtyping import Float
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch import Tensor

import wandb
from configs.bbf import EnvironmentConfig, NetworkConfig, TrainingConfig
from models import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from utils import TargetNetworkUpdater, \
    exponential_scheduler, linearly_decaying_epsilon


def make_env(env_id, seed, idx, capture_video=False, run_name=None):
    def thunk() -> gym.Env:
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env
    return thunk


def project_distribution(
        weights: Float[Tensor, "batch n_atoms"],
        supports: Float[Tensor, "batch n_atoms"],
        target_support: Float[Tensor, "n_atoms"],
) -> Float[Tensor, "batch n_atoms"]:
    """Projects a batch of (support, next_probabilities) onto target_support.
    Code is adapted from dopamine Rainbow agent
    Based on equation (7) in (Bellemare et al., 2017): https://arxiv.org/abs/1707.06887

    Args:
        weights: Tensor of shape (batch_size, num_dims) defining weights on the
          original support points. Although for the CategoricalDQN agent these
          weights are probabilities, it is not required that they are.
        supports: Tensor of shape (batch_size, num_dims) defining supports for the
          distribution.
        target_support: Tensor of shape (num_dims) defining support of the projected
          distribution. The values must be monotonically increasing. Vmin and Vmax
          will be inferred from the first and last elements of this tensor,
          respectively. The values in this tensor must be equally spaced.

    Returns:
        A Tensor of shape (batch_size, num_dims) with the projection of a batch of
        (support, weights) onto target_support.
    """
    target_support_deltas = target_support[1:] - target_support[:-1]
    delta_z = target_support_deltas[0]

    v_min, v_max = target_support[0], target_support[-1]
    batch_size, n_atoms = supports.shape
    clipped_support = torch.clamp(supports, v_min, v_max)[:, None, :]
    tiled_support = clipped_support.repeat(1, 1, n_atoms, 1)

    reshaped_target_support = einops.repeat(target_support, 'n_atoms -> batch n_atoms 1', batch=batch_size)

    numerator = torch.abs(tiled_support - reshaped_target_support)
    quotient = 1 - (numerator / delta_z)

    clipped_quotient = torch.clamp(quotient, 0, 1)
    weights = einops.rearrange(weights, 'batch n_atoms -> batch 1 n_atoms')
    inner_prod = clipped_quotient * weights

    projection = torch.sum(inner_prod, dim=3)
    projection = projection.view(batch_size, n_atoms)

    return projection


class BBFAgent(torch.nn.Module):
    def __init__(self,
                 network=DQN,
                 v_min=-10,
                 v_max=10,
                 n_atoms=51
                 ):
        super(BBFAgent, self).__init__()

        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, n_atoms)
        self.n_atoms = n_atoms

        # Store configurations
        self.env_config = EnvironmentConfig()
        self.network_config = NetworkConfig()
        self.train_config = TrainingConfig()

        seed = 0
        # Environment setup
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(self.env_config.env_name, seed + i, i, False) for i in range(1)]
        )
        self.n_actions = self.envs.single_action_space.n
        sample_observations, _ = self.envs.reset()

        # Networks
        observation_shape = sample_observations.shape[1:]
        self.online_network = network(observation_shape, self.n_actions, self.n_atoms)
        self.target_network = network(observation_shape, self.n_actions, self.n_atoms)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.ema_updater = TargetNetworkUpdater(self.online_network, self.target_network, self.network_config.tau)

        self.optimizer = torch.optim.AdamW(params=self.online_network.parameters(),
                                           lr=self.network_config.learning_rate,
                                           weight_decay=self.network_config.weight_decay)
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.network_config.buffer_size,
            observation_space=self.envs.single_observation_space,
            action_space=self.envs.single_action_space,
            device='cpu',
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

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

    def select_epsilon_greedy_action(
            self,
            observations: Float[Tensor, 'batch c h w'],
            epsilon: float,
    ) -> np.array:
        if random.random() < epsilon:
            return np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        else:
            observations = torch.tensor(np.array(observations), dtype=torch.float32)
            action_probs = self.target_network(observations)
            q_values = einops.einsum(action_probs, self.support, 'envs n_actions n_atoms, n_atoms -> envs n_actions')
            return q_values.argmax(dim=-1).numpy()

    @torch.no_grad()
    def compute_target(
            self,
            next_observations: Float[Tensor, 'batch c h w'],
            rewards: Float[Tensor, 'batch'],
            dones: Float[Tensor, 'batch'],
            gamma: int,
            distributional: bool,
    ) -> Float[Tensor, 'batch n_atoms']:
        if distributional:
            probabilities = self.target_network(next_observations)
            q_values = einops.einsum(probabilities, self.support, 'batch n_actions n_atoms, n_atoms -> batch n_actions')
            best_actions = torch.argmax(q_values, dim=-1)
            batch_size = next_observations.shape[0]
            next_probabilities = probabilities[torch.arange(batch_size), best_actions]

            support = einops.repeat(self.support, 'n_atoms -> batch n_atoms', batch=batch_size)
            target_support = rewards + gamma * support * (1 - dones)
            target = project_distribution(next_probabilities, target_support, self.support)
        else:
            next_q_values = self.target_network(next_observations).max(-1)[0]
            target = rewards + gamma * next_q_values * (1 - dones)
        return target

    def compute_loss(
            self,
            batch: ReplayBufferSamples,
            update_horizon: int,
            gamma: int
    ):
        batch_size = batch.observations.shape[0]

        target = self.compute_target(batch.next_observations, batch.rewards, batch.dones, gamma, distributional=True)

        distributional = True
        if distributional:
            probabilities = self.online_network(batch.observations)[torch.arange(batch_size), batch.actions.squeeze()]
            loss = torch.nn.functional.cross_entropy(probabilities, target)
        else:
            q_values = self.online_network(batch.observations)[torch.arange(batch_size), batch.actions.squeeze()]
            loss = torch.nn.functional.huber_loss(q_values, target, delta=1.0)
        return loss

    def train(self, project_name="bbf", run_name=None, disable_wandb=False):
        wandb.init(
            project=project_name,
            mode="disabled" if disable_wandb else "online",
            name=run_name,
        )

        observations, _ = self.envs.reset()

        for step in range(self.train_config.num_steps):
            epsilon = linearly_decaying_epsilon(
                decay_period=self.train_config.epsilon_decay_period,
                step=step,
                warmup_steps=self.network_config.min_replay_history,
                epsilon=self.train_config.epsilon_train
            )
            actions = self.select_epsilon_greedy_action(observations, epsilon)
            next_observations, rewards, dones, _, infos = self.envs.step(actions)
            self.replay_buffer.add(observations, next_observations, actions, rewards, dones, infos)

            if 'final_info' in infos:
                for info in infos['final_info']:
                    if info and "episode" in info:
                        print(f"Step={step}, Episode Reward={info['episode']['r']}")
                        wandb.log({"episode_reward": info['episode']['r']})
                        wandb.log({"episode_length": info['episode']['l']})

            observations = next_observations

            if step > self.network_config.min_replay_history:
                for _ in range(self.train_config.replay_ratio):
                    self.train_step()

            self.ema_updater.soft_update()

        wandb.finish()
        print("Training completed!")

    def train_step(self):
        gradient_steps_since_reset = self.gradient_steps % self.train_config.reset_interval
        batch = self.replay_buffer.sample(self.network_config.batch_size)
        loss = self.compute_loss(
            batch=batch,
            update_horizon=self.update_horizon_scheduler(gradient_steps_since_reset),
            gamma=self.gamma_scheduler(gradient_steps_since_reset),
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.network_config.max_gradient_norm)
        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % 100 == 0:
            wandb.log({"loss": loss.item(), "gradient_steps": self.gradient_steps})
        if self.gradient_steps % self.train_config.reset_interval == 0:
            self.shrink_and_perturb_parameters()


if __name__ == '__main__':
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic
    agent = BBFAgent()
    agent.train(disable_wandb=True)
