import torch
from torch import nn


class ConditionalTransformer(nn.Module):
    def __init__(self, num_channels, hidden_channels, n_heads, num_layers, activation):
        super().__init__()

        # d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

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
        input = torch.cat([inp, catted], dim=-1)

        for transformer in self.layers:
            x = transformer(input)
        x = self.proj(x[:, conditioning.shape[1] :, :])

        """
        input = torch.cat([conditioning, x], dim=-1)
        x = self.transformer(input)[:, self.conditioning_size:]
        """
        return x[:, :, 0]
