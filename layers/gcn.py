import torch
from torch import nn, Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import  remove_self_loops

class WeightedGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,cached: bool = False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        # 定义可学习参数
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.cached_result = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        edge_index = remove_self_loops(edge_index=edge_index)[0]
        # _, edge_attr = add_remaining_self_loops(edge_index, fill_value=1, num_nodes=x.shape[0])

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            False, True, self.flow, x.dtype)

        # 消息传播与聚合
        return self.lin(self.propagate(edge_index, x=x, edge_weight=edge_weight))

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # 加权消息：x_j * edge_weight
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps.item()})"