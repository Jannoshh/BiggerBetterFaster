import einops
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_shape, n_actions):
        super(DQN, self).__init__()
        channels = state_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(state_shape)
        self.fc = nn.Linear(conv_out_size, n_actions)

    def _get_conv_out(self, shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        conv = self.conv(x)
        x = einops.rearrange(conv, 'b c h w -> b (c h w)')
        return self.fc(x)


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
