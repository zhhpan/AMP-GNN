from typing import Callable

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor


class WeightedGINConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mlp_func: Callable,
            eps: float = 0.,
            aggr: str = "add",
            bias: bool = True,
            **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        # 定义MLP（通过工厂函数构造）
        self.mlp = mlp_func(in_channels=in_channels,
                            out_channels=out_channels,
                            bias = bias)

        self.eps = Parameter(torch.Tensor([eps]))

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            edge_weight: OptTensor = None,
            edge_attr: OptTensor = None
    ) -> Tensor:
        # 消息传播（聚合邻居信息）
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            edge_attr=edge_attr
        )
        # 应用GIN公式：(1 + eps) * x + 聚合结果
        out = (1 + self.eps) * x + out

        # 通过MLP变换
        return self.mlp(out)

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

