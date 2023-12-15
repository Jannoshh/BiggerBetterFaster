import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor


class DQN(nn.Module):
    def __init__(self,
                 state_shape,
                 n_actions,
                 n_atoms,
                 dropout=0.0,
                 hidden_units=512,
                 dueling=True,
                 ):
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.dueling = dueling

        self.encoder = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.Dropout2d(dropout),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Dropout2d(dropout),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout2d(dropout),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
        )

        flattened_dim = self._get_flattened_dim(state_shape)

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, hidden_units),
            nn.ReLU(),
        )

        if self.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_units, n_atoms),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_units, n_actions * n_atoms),
            )
        else:
            self.advantage_stream = nn.Linear(hidden_units, n_actions * n_atoms)

    def _get_flattened_dim(self, shape):
        out = self.encoder(torch.zeros(1, *shape))
        return out.shape[-1]

    def forward(self, state: Float[Tensor, 'batch c h w']) -> Float[Tensor, 'batch n_actions n_atoms']:
        latent = self.encoder(state)
        latent = self.fc(latent)
        if self.dueling:
            value = self.value_stream(latent)
            advantage = self.advantage_stream(latent)
            value = einops.rearrange(value, 'b n_atoms -> b 1 n_atoms')
            advantage = einops.rearrange(advantage, 'b (n_actions n_atoms) -> b n_actions n_atoms', n_actions=self.n_actions)
            logits = value + advantage - einops.reduce(advantage, 'b n_actions n_atoms -> b 1 n_atoms', reduction='mean')
        else:
            logits = self.advantage(latent)
            logits = einops.rearrange(logits, 'b (n_actions n_atoms) -> b n_actions n_atoms', n_actions=self.n_actions)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities


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
