from typing import Callable

import torch
from torch import nn
from torch.nn import ReLU
from torch_geometric.typing import OptTensor

from layers.load_helper import get_component_list


class ActionNetwork(nn.Module):
    """动作网络"""

    def __init__(self, in_dim: int, model_type: str, hidden_dim: int, num_layers: int, dropout: float, mlp_func: Callable, device):
        super().__init__()
        self.device = device
        self.net = get_component_list(in_dim=in_dim, out_dim=2, hidden_dim=hidden_dim, num_layers=num_layers,model_type=model_type, mlp_func=mlp_func, device=self.device)
        self.dropout = nn.Dropout(dropout)
        self.act_func = ReLU()
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, env_edge_attr: OptTensor, act_edge_attr: OptTensor) -> torch.Tensor:
        edge_attrs = [env_edge_attr] + (self.num_layers - 1) * [act_edge_attr]
        # 计算边特征
        for idx, (edge_attr, layer) in enumerate(zip(edge_attrs[:-1], self.net[:-1])):
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.dropout(x)
            x = self.act_func(x)
        x = self.net[-1](x, edge_index=edge_index, edge_attr=edge_attrs[-1])
        return x