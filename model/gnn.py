import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor

import torch_geometric
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import conv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax


class e3GATAttendOnlyConv(conv.MessagePassing):
    def __init__(
        self,
        channels,
        heads,
        e3dims=3,
        bias=False,
        negative_slope=0.2,
        dropout=0.0,
        add_self_loops=True,
        **kwargs
    ):
        super().__init__(node_dim=0, **kwargs)
        self.channels = channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.e3dims = e3dims

        # attention doesnt use coordinates, only uses other features + distance (- e3dims + 1 distance)
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

        # build adjacency matrix from edge_index
        num_nodes = x.size(0)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1.0

        if self.add_self_loops:
            adj_matrix[edge_index[0], edge_index[0]] = 1.0

        #
        # simulate receptive field by gnn at each diffusion step
        #
        walks = torch.linalg.matrix_power(adj_matrix, step)
        s, t = torch.nonzero(walks, as_tuple=True)
        edge_attr = walks[s, t]
        new_edge_index = torch.stack([s, t], dim=0)

        #
        # gat
        #
        alpha = self.edge_updater(new_edge_index, x=x, edge_attr=edge_attr)
        # propogate shouldn't use coordinate parameter
        out = self.propagate(new_edge_index, x=x[:, self.e3dims :], alpha=alpha)
        # average over all attention heads
        out = out.mean(dim=1)

        #
        # e3 gnn
        #
        source_features = x[new_edge_index[1], : self.e3dims]
        neighbor_features = x[new_edge_index[0], : self.e3dims]
        # compute and average attention over all heads
        alpha_scaled_features = (source_features - neighbor_features).view(
            neighbor_features.size(0), 1, -1
        ) * alpha.view(alpha.size(0), alpha.size(1), 1)
        alpha_scaled_features = alpha_scaled_features.mean(dim=1)
        num_nodes = x.size(0)
        # initialize coordinates with initial coordinates
        aggregated_features = x[:, : self.e3dims]
        new_contributions = torch.zeros_like(aggregated_features)
        # count num of contributions
        zeros = torch.zeros(num_nodes) - 1
        zeros.index_add_(0, new_edge_index[1], torch.ones(new_edge_index.size(1)))
        # aggregate diffs in coordinates times attention
        new_contributions.index_add_(0, new_edge_index[1], alpha_scaled_features)
        # combine and scale by num of contributions
        aggregated_features += new_contributions / zeros.view(-1, 1)

        if self.bias is not None:
            out = out + self.bias

        # add coordinates back to output
        out = torch.cat([aggregated_features, out], dim=-1)

        return out

    def message(self, x_j, alpha):
        # x_j does not contain coordinates

        # multiply by attention
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
        # x_j and x_i contain coordinates

        # aggregate
        x = x_i + x_j
        dists = torch.sqrt(
            (x_i[:, : self.e3dims] - x_j[:, : self.e3dims]).pow(2).sum(axis=-1)
        ).view(-1, 1)
        x = torch.cat([x[:, self.e3dims :], dists], axis=-1)

        # now they don't

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            x = x + edge_attr

        # attention
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x.view(x.size(0), 1, x.size(1)) * self.att).sum(dim=-1)
        # softmax across neighbors
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha


class e3GATAttend(torch_geometric.nn.models.GCN):
    # edge weights are number of diffusion steps, hack to pass to conv forward
    supports_edge_weight = True
    supports_edge_attr = False

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        e3dims: int = 3,
        heads: int = 3,
        **kwargs
    ) -> e3GATAttendOnlyConv:
        # first e3dims are coordinates
        assert in_channels == out_channels
        return e3GATAttendOnlyConv(in_channels, heads=heads, e3dims=e3dims, **kwargs)
