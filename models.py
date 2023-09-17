import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class DQN(nn.Module):
    def __init__(self, state_shape, n_actions, channels=[32, 64, 64], kernel_sizes=[8, 4, 3], strides=[4, 2, 1],
                 hidden_units=512):
        super(DQN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(state_shape[0], channels[0], kernel_size=kernel_sizes[0], stride=strides[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[1], stride=strides[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=kernel_sizes[2], stride=strides[2]),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
        )

        flattened_dim = self._get_flattened_dim(state_shape)

        self.q_learning_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_actions),
        )

    def _get_flattened_dim(self, shape):
        out = self.encoder(torch.zeros(1, *shape))
        return out.shape[-1]

    def forward(self, state):
        latent = self.encoder(state)
        q_values = self.q_learning_head(latent)
        return q_values


class ImpalaCNN(nn.Module):
    def __init__(self,
                 width_scale: int = 1,
                 dims: tuple[int] = (16, 32, 32),
                 num_blocks: int = 2,
                 dtype: torch.dtype = torch.float32,
                 dropout: float = 0.0,
                 initializer=torch.nn.init.xavier_uniform_):
        super(ImpalaCNN, self).__init__()

        self.width_scale = width_scale
        self.dims = dims
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.dropout = dropout
        self.initializer = initializer

        in_dim = 1  # Initial input dimension
        self.residual_stages = torch.nn.ModuleList()
        for width in self.dims:
            out_dim = int(width * self.width_scale)
            self.residual_stages.append(
                ResidualStage(
                    in_dims=in_dim,
                    dims=out_dim,
                    num_blocks=self.num_blocks,
                    dtype=self.dtype,
                    dropout=self.dropout,
                    initializer=self.initializer
                )
            )
            in_dim = out_dim

    def forward(self, x):
        for residual_stage in self.residual_stages:
            x = residual_stage(x)
        x = torch.nn.functional.relu(x)
        return x


class ResidualStage(nn.Module):
    """A single residual stage for an Impala-style ResNet."""

    def __init__(self, in_dims: int, dims: int, num_blocks: int, use_max_pooling: bool = True,
                 dtype: torch.dtype = torch.float32, dropout: float = 0.0,
                 initializer=nn.init.xavier_uniform_):
        super(ResidualStage, self).__init__()

        self.dims = dims
        self.num_blocks = num_blocks
        self.use_max_pooling = use_max_pooling
        self.dtype = dtype
        self.dropout = dropout
        self.initializer = initializer

        # Define initial convolution layer
        self.conv_initial = nn.Conv2d(in_channels=in_dims, out_channels=dims, kernel_size=3,
                                      stride=1, padding=1, bias=False)
        self.initializer(self.conv_initial.weight)

        # Define blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            block = nn.Sequential(
                nn.ReLU(),
                nn.Dropout2d(self.dropout),
                nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.blocks.append(block)

    def forward(self, x):
        x = self.conv_initial(x)
        if self.use_max_pooling:
            x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            residual = x
            x = block(x)
            x += residual
        return x


if __name__ == '__main__':
    model = ImpalaCNN()
    x = torch.randn(1, 1, 84, 84)  # Example input
    output = model(x)
    print(output.shape)
