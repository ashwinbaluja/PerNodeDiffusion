import torch
from torch import nn

from model.utils import SinusoidalEmbedding, MolecularEmbedding
from model.gnn import e3GATAttend
from model.transformer import ConditionalTransformer


class DiffusionStep(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_channels: int,
        n_heads: int = 6,
        num_layers: int = 12,
        time_emb_size: int = 16,
        max_mol: int = 118,
        mol_emb_size: int = 16,
        activation: str = "relu",
        dropout: float = 0.0,
        e3dims: int = 3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.time_emb_size = time_emb_size
        self.mol_emb_size = mol_emb_size

        self.activation = activation

        self.sinusoidal_embedding = SinusoidalEmbedding(time_emb_size)
        self.molecular_embedding = MolecularEmbedding(max_mol, mol_emb_size)

        self.conditioning_size = time_emb_size + mol_emb_size

        self.gnn = e3GATAttend(
            in_channels=num_channels,
            hidden_channels=num_channels,
            heads=n_heads,
            num_layers=1,
            dropout=dropout,
            e3dims=e3dims,
        )

        self.transformer = ConditionalTransformer(
            num_channels,
            hidden_channels,
            self.conditioning_size,
            n_heads,
            num_layers,
            activation=activation,
        )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for module in self.transformer.modulation:
            # module.weight.data.one_()
            # module.bias.data.zero_()
            pass

    def forward(
        self,
        x,
        conditioning,
        edge_index,
        diffusion_time=torch.tensor(1),
        gnn_time_step=1,
        edge_weight=None,
    ):

        time_emb = self.sinusoidal_embedding(diffusion_time)  # (self.time_emb_size)

        molecules = self.molecular_embedding(
            conditioning
        )  # (nodes x self.mol_emb_size)

        conditioning = torch.cat([time_emb, molecules], dim=-1)

        x = self.transformer(x, conditioning)
        x = self.gnn(x, edge_index, edge_weight=gnn_time_step)
        return x
