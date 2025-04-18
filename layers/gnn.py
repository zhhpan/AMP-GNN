import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor


class WeightedGNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, aggr: str, bias: bool, **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(2 * in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, edge_attr=edge_attr)
        out = self.lin(torch.cat((x, out), dim=-1))
        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor = None) -> Tensor:
        if edge_attr is None:
            if edge_weight is None:
                return x_j
            else:
                return edge_weight.view(-1, 1) * x_j
        else:
            if edge_weight is None:
                return x_j + edge_attr
            else:
                return edge_weight.view(-1, 1) * (x_j + edge_attr)
