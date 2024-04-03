import torch
from torch import nn


# does tabular data need positional encodings?
class ConditionalTransformer(nn.Module):
    def __init__(self, num_channels, hidden_channels, n_heads, num_layers, activation):
        super().__init__()

        self.num_channels = num_channels
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.activation = activation

        self.layers = nn.ModuleList()

        self.input_size = hidden_channels

        self.proj = nn.Linear(hidden_channels, 1)

        for i in range(self.num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.input_size,
                    nhead=self.n_heads,
                    dim_feedforward=self.input_size,
                    activation=self.activation,
                    norm_first=True,
                )
            )

    def forward(self, x, conditioning):
        inp = torch.zeros(
            (x.shape[0], conditioning.shape[1] + x.shape[1], self.input_size - 1),
            device=x.device,
            dtype=x.dtype,
        )
        catted = torch.cat([conditioning, x], dim=-1)[:, :, None]
        # iffy about this - kind of like one hot, but we just put the value in the 0 index of each
        # token embedding, and let the transformer learn the rest
        input = torch.cat([inp, catted], dim=-1)

        for transformer in self.layers:
            x = transformer(input)
            # teacher forcing? yes or no? (force conditioning to be the same)
        x = self.proj(x[:, conditioning.shape[1] :, :])
        # any uses for an additional last token? maybe get some useful information out of it somehow, this probably learns a nice embedding representation of each atom(!!)

        return x[:, :, 0]
