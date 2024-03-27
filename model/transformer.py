import torch
from torch import nn


class ConditionalTransformer(nn.Module):
    def __init__(
        self, num_channels, conditioning_size, n_heads, num_layers, dropout, activation
    ):
        super().__init__()

        self.num_channels = num_channels
        self.conditioning_size = conditioning_size
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        self.layers = nn.ModuleList()

        input_size = num_channels + conditioning_size

        for i in range(self.num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=input_size,
                    nhead=self.n_heads,
                    dim_feedforward=input_size,
                    dropout=self.dropout,
                    activation=self.activation,
                    norm_first=True,
                )
            )

    def forward(self, x, conditioning):
        for i in self.layers:
            input = torch.cat([conditioning, x], dim=-1)
            x = i(input)[:, self.conditioning_size :]

        return x
