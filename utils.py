import einops
import numpy as np
import torch


def preprocess_state(state):
    state = state / 255.0
    if len(state.shape) == 3:
        return einops.rearrange(state, "h w c -> c h w")
    else:
        return einops.rearrange(state, "h w -> 1 h w")


class TargetNetworkUpdater:
    def __init__(self, source_network, target_network, tau):
        self.source_network = source_network
        self.target_network = target_network
        self.tau = tau

    def soft_update(self):
        for target_param, source_param in zip(self.target_network.parameters(), self.source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


def exponential_scheduler(
    decay_period, initial_value, final_value, warmup_steps=0,
):
    """Instantiate a logarithmic schedule for a parameter.

    Args:
        decay_period: float, the period over which the value is decayed.
        initial_value: float, the starting value for the parameter.
        final_value: float, the final value for the parameter.
        warmup_steps: int, the number of steps taken before decay starts.

    Returns:
        A function mapping step to parameter value.
    """

    start = torch.log(torch.tensor(initial_value))
    end = torch.log(torch.tensor(final_value))

    if decay_period == 0:
        return lambda x: initial_value if x < warmup_steps else final_value

    def scheduler(step):
        step = torch.tensor(step, dtype=torch.float32)
        steps_left = decay_period + warmup_steps - step
        bonus_frac = steps_left / decay_period
        bonus = torch.clamp(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = torch.exp(new_value)
        return new_value.item()

    return scheduler


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.

    Args:
        decay_period: float, the period over which epsilon is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before epsilon is decayed.
        epsilon: float, the final value to which to decay the epsilon parameter.

    Returns:
        A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus

# Test the function
epsilon_values = [linearly_decaying_epsilon(10000, i, 1000, 0.1) for i in range(0, 11000, 1000)]
epsilon_values
