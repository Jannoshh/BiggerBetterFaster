import cv2
import einops
import torch


def preprocess_state(state):
    # Convert state image to grayscale, resize and normalize
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    state = state / 255.0
    return einops.rearrange(state, "h w -> 1 h w")


def epsilon_decay(frame_idx, epsilon_start, epsilon_end, epsilon_decay_duration):
    return max(epsilon_end, epsilon_start - frame_idx / epsilon_decay_duration)


class TargetNetworkUpdater:
    def __init__(self, source_network, target_network, tau):
        self.source_network = source_network
        self.target_network = target_network
        self.tau = tau

    def soft_update(self):
        for target_param, source_param in zip(self.target_network.parameters(), self.source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


def shrink_and_perturb_parameters(model, alpha):
    for name, param in model.named_parameters():
        if 'encoder' in name:  # Assuming 'encoder' is part of the name for encoder layers
            phi = torch.randn_like(param).normal_(0, 0.01)  # Assuming a normal initializer with mean 0 and std 0.01
            param.data = alpha * param.data + (1 - alpha) * phi
        elif 'final' in name:
            phi = torch.randn_like(param).normal_(0, 0.01)  # Assuming a normal initializer with mean 0 and std 0.01
            param.data = phi


def exponential_decay_scheduler(
    decay_period, warmup_steps, initial_value, final_value
):
    """Instantiate a logarithmic schedule for a parameter.

    Args:
        decay_period: float, the period over which the value is decayed.
        warmup_steps: int, the number of steps taken before decay starts.
        initial_value: float, the starting value for the parameter.
        final_value: float, the final value for the parameter.

    Returns:
        A decay function mapping step to parameter value.
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

