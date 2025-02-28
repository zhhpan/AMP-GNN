from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor


class WeightedGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=bias)


    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
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
