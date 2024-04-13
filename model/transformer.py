import torch
from torch import nn


# does tabular data need positional encodings?
class ConditionalTransformer(torch.nn.Module):
    def __init__(
        self,
        num_channels,
        hidden_channels,
        conditioning_size,
        n_heads,
        num_layers,
        activation,
    ):
        super().__init__()

        # d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

        self.num_channels = num_channels
        self.conditioning_size = conditioning_size
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.activation = activation

        self.layers = torch.nn.ModuleList()

        self.input_size = hidden_channels

        self.linear = torch.nn.Linear(num_channels, hidden_channels)
        self.proj = torch.nn.Linear(hidden_channels, 1)

        for i in range(self.num_layers):
            self.layers.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=self.input_size,
                    nhead=self.n_heads,
                    dim_feedforward=self.input_size,
                    activation=self.activation,
                    norm_first=True,
                )
            )

    def forward(self, x, conditioning):
        inp = torch.zeros(
            (x.shape[0], x.shape[1] + conditioning.shape[1], self.input_size - 2),
            device=x.device,
            dtype=x.dtype,
        )
        catted = torch.cat([conditioning, x], dim=-1)
        positional = (
            torch.linspace(0, 1, catted.shape[1] + 1, device=x.device, dtype=x.dtype)[
                1:
            ]
            .view(1, -1, 1)
            .expand(x.shape[0], -1, -1)
        )
        x = torch.cat([catted[:, :, None], positional, inp], dim=-1)
        for transformer in self.layers:
            x = transformer(x)
        x = self.proj(x)[:, conditioning.shape[1] :, 0]
        """
        input = torch.cat([conditioning, x], dim=-1)
        x = self.transformer(input)[:, self.conditioning_size:]
        """
        return x
