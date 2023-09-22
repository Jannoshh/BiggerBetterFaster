from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    env_name: str = 'ALE/BattleZone-v5'
    wandb_project: str | None = "bbf"
    wandb_name: str | None = None


@dataclass
class NetworkConfig:
    buffer_size: int = 10000
    min_replay_history = 2000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    tau: float = 0.005
    max_gradient_norm: float = 10.0


@dataclass
class TrainingConfig:
    num_steps: int = 100000
    epsilon_train: float = 0
    epsilon_decay_period: int = 2001
    replay_ratio: int = 1
    frames_stack: int = 4
    action_repeat: int = 4
    perturbation_interval: int = 100
    reset_interval: int = 40000
    alpha: float = 0.5
    min_update_horizon: int = 3
    max_update_horizon: int = 10
    min_gamma: float = 0.97
    max_gamma: float = 0.997
    cycle_steps: int = 10000
