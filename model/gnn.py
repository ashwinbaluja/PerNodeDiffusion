import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor

import torch_geometric
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import conv
from torch_geometric.typing import Adj, OptTensor


class e3GATAttendOnlyConv(conv.MessagePassing):
    def __init__(
        self,
        channels,
        heads,
        e3dims=3,
        bias=False,
        negative_slope=0.2,
        dropout=0.0,
        **kwargs
    ):
        super().__init__(node_dim=0, **kwargs)
        self.channels = channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.e3dims = e3dims

        self.att = Parameter(torch.empty(1, heads, channels - e3dims + 1))

        if bias:
            self.bias = Parameter(torch.empty(channels - e3dims))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight=1):
        # edge_weight is the number of diffusion steps! (not the weight)
        step = edge_weight
        H, C = self.heads, self.channels

        num_nodes = x.size(0)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0

        # simulate receptive field by gnn at each diffusion step
        walks = torch.linalg.matrix_power(adj_matrix, step)
        src, tgt = torch.nonzero(walks, as_tuple=True)
        edge_attr = walks[src, tgt]
        new_edge_index = torch.stack([src, tgt], dim=0)

        none3 = x[:, self.e3dims :]

        alpha = self.edge_updater(new_edge_index, x=x, edge_attr=edge_attr)
        out = self.propagate(new_edge_index, x=none3, alpha=alpha)

        # average over all attention heads
        out = out.mean(dim=1)

        # e3 gnn
        neighbor_features = x[new_edge_index[0], : self.e3dims]
        # compute and average attention over all heads
        alpha_scaled_features = neighbor_features.view(
            neighbor_features.size(0), 1, -1
        ) * alpha.view(alpha.size(0), alpha.size(1), 1)
        alpha_scaled_features = alpha_scaled_features.mean(dim=1)
        num_nodes = x.size(0)
        aggregated_features = torch.zeros(
            (num_nodes, self.e3dims), device=x.device, dtype=x.dtype
        )
        aggregated_features.index_add_(0, new_edge_index[1], alpha_scaled_features)

        if self.bias is not None:
            out = out + self.bias

        out = torch.cat([aggregated_features, out], dim=-1)

        return out

    def message(self, x_j, alpha):
        return alpha.view(alpha.size(0), alpha.size(1), 1) * x_j.view(
            x_j.size(0), 1, -1
        )

    def edge_update(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: int,
    ) -> Tensor:
        # aggregate
        x = x_i + x_j
        dists = torch.sqrt(
            (x_i[:, : self.e3dims] - x_j[:, : self.e3dims]).pow(2).sum(axis=-1)
        ).view(-1, 1)
        x = torch.cat([x[:, self.e3dims :], dists], axis=-1)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            x = x + edge_attr

        # attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x.view(x.size(0), 1, x.size(1)) * self.att).sum(dim=-1)
        alpha = torch_geometric.utils.softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha


class e3GATAttend(torch_geometric.nn.models.GCN):
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        e3dims: int = 3,
        heads: int = 3,
        **kwargs
    ) -> conv.MessagePassing:
        # first e3dims are coordinates
        assert in_channels == out_channels
        return e3GATAttendOnlyConv(in_channels, heads=heads, e3dims=e3dims, **kwargs)
