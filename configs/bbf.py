from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    env_name: str = 'ALE/BattleZone-v5'
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


@dataclass
class NetworkConfig:
    buffer_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    tau: float = 0.005

@dataclass
class TrainingConfig:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 10000
    num_steps: int = 100000
    epsilon_decay_last_frame: int = 100000
    replay_ratio: int = 4
    perturbation_interval: int = 100
    reset_interval: int = 40000
    alpha: float = 0.5