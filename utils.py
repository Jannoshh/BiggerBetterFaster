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
