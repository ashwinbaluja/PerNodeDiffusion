import torch
from torch import nn

from model.utils import SinusoidalEmbedding, MolecularEmbedding
from model.transformer import ConditionalTransformer
from model.gnn import e3GATAttend


class DiffusionStep(nn.Module):
    def __init__(
        self,
        num_channels: int,
        n_heads: int = 6,
        num_layers: int = 12,
        time_emb_size: int = 32,
        mol_emb_size: int = 118,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.time_emb_size = time_emb_size
        self.mol_emb_size = mol_emb_size

        self.activation = activation

        self.sinusoidal_embedding = SinusoidalEmbedding(time_emb_size)
        self.molecular_embedding = MolecularEmbedding(mol_emb_size)

        self.conditioning_size = time_emb_size + mol_emb_size

        self.gnn = e3GATAttend(
            in_channels=num_channels,
            hidden_channels=num_channels,
            heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.transformer = ConditionalTransformer(
            num_channels,
            self.conditioning_size,
            n_heads,
            num_layers,
            activation=activation,
        )

    def forward(
        self,
        x,
        conditioning,
        edge_index,
        diffusion_time=torch.tensor(1),
        gnn_time_step=1,
    ):
        # diffusion_time is from the scheduler
        # gnn_time_step is the real number step out of the total number of steps
        # OR
        # gnn_time_step is the same as from the scheduler. as of yet to be determined

        time_emb = self.sinusoidal_embedding(diffusion_time)  # (self.time_emb_size)

        molecules = self.molecular_embedding(
            conditioning
        )  # (nodes x self.mol_emb_size)

        conditioning = torch.cat(
            [
                time_emb.view(1, -1).expand(molecules.size(0), self.time_emb_size),
                molecules,
            ],
            dim=-1,
        )

        x = self.transformer(x, conditioning)
        x = self.gnn(x, edge_index, edge_weight=gnn_time_step)

        return x
