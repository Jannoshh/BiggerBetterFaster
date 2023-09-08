import cv2
import einops


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

