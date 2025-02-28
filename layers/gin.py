from typing import Callable
import torch
from torch import nn, Tensor
from torch.nn import Linear, Dropout, GELU, Parameter, ReLU, Module, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor


class WeightedGINConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mlp_func: Callable,
            eps: float = 0.,
            train_eps: bool = True,
            aggr: str = "add",
            bias: bool = True,
            **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        # 定义MLP（通过工厂函数构造）
        self.mlp = mlp_func(in_channels=in_channels,
                            out_channels=out_channels,
                            bias = bias)

        # 可学习的epsilon参数（中心节点权重）
        self.initial_eps = eps
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()
        self.lin = Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

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