from typing import Callable

import torch

from torch import nn, Tensor
from torch.cuda import device
from torch.nn import Linear, Dropout, GELU, Parameter, ReLU, Module, ModuleList
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import  remove_self_loops


class TempSoftPlus(Module):
    def __init__(self, learn_temp: bool,
                       temp_model_type: str,
                       tau0: float,
                       temp: float,
                       gin_mlp_func: Callable,
                       env_dim: int,
                       device: device):
        super(TempSoftPlus, self).__init__()
        model_list =get_component_list(in_dim=env_dim, hidden_dim=env_dim, out_dim=1, num_layers=1,
                                                           bias=False,model_type=temp_model_type,
                                                           mlp_func=gin_mlp_func,device=device)
        self.linear_model = ModuleList(model_list)
        self.softplus = nn.Softplus(beta=1)
        self.tau0 = tau0

    def forward(self, x: Tensor, edge_index: Adj):
        x = self.linear_model[0](x=x, edge_index=edge_index)
        x = self.softplus(x) + self.tau0
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float('inf'), 0.)

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


def get_component_list(in_dim, out_dim, hidden_dim, num_layers, model_type, mlp_func,device,bias = True):
    """
    获取组件列表
    :param device: 设备位置
    :param model_type: 组件类型
    :param in_dim: 输入层维度
    :param out_dim: 输出层维度
    :param hidden_dim: 隐藏层维度
    :param num_layers: 组件层数
    :param mlp_func: GIN所需要的多层感知机的函数
    :return: 组件列表
    """
    component_list = []
    dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    if model_type == 'GIN':
        for i in range(len(dim_list) - 1):
            component_list.append(WeightedGINConv(in_channels = dim_list[i], out_channels = dim_list[i + 1], mlp_func = mlp_func, bias=bias).to(device))
    elif model_type == 'GCN':
        for i in range(len(dim_list) - 1):
            component_list.append(WeightedGCNConv(in_channels = dim_list[i], out_channels = dim_list[i + 1], mlp_func = mlp_func, bias=bias).to(device))
    return component_list


class ActionNetwork(nn.Module):
    """动作网络"""

    def __init__(self, in_dim: int, model_type: str, hidden_dim: int, num_layers: int, dropout: float, mlp_func: Callable, device):
        super().__init__()
        self.device = device
        self.net = get_component_list(in_dim=in_dim, out_dim=4, hidden_dim=hidden_dim, num_layers=num_layers,model_type=model_type, mlp_func=mlp_func, device=self.device)
        self.dropout = nn.Dropout(dropout)
        self.act_func = ReLU()
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 计算边特征
        for idx, layer in enumerate(self.net[:-1]):
            x = layer(x, edge_index=edge_index)
            x = self.dropout(x)
            x = self.act_func(x)
        x = self.net[-1](x, edge_index=edge_index)
        return x



class EnvironmentNetwork(nn.Module):
    """环境网络"""

    def __init__(self,
                 in_dim: int,
                 env_dim: int,
                 out_dim: int,
                 num_layers: int,
                 dropout: float,
                 model_type: str,
                 mlp_func: callable,
                 device: torch.device):
        super().__init__()

        # 编码器 使用线性层
        self.encoder = nn.Linear(in_dim, env_dim)

        # 组件
        self.component = nn.ModuleList(get_component_list(in_dim = env_dim,
                                                          out_dim = env_dim,
                                                          hidden_dim = env_dim,
                                                          num_layers = num_layers,
                                                          model_type = model_type,
                                                          mlp_func = mlp_func,
                                                          device = device))

        # 解码器
        self.decoder = nn.Sequential(Linear(env_dim, out_dim), Dropout(dropout), GELU())




class CoGNN(nn.Module):

    def __init__(self,
                 gumbel_params: dict,  # 单独参数
                 env_params: dict,
                 action_params: dict,
                 device: torch.device):
        super().__init__()

        # 初始化子网络
        self.env_net = EnvironmentNetwork(
            in_dim=env_params['in_dim'],
            env_dim=env_params['env_dim'],
            out_dim=env_params['out_dim'],
            num_layers=env_params['num_layers'],
            dropout=env_params['dropout'],
            mlp_func=gumbel_params['gin_mlp_func'],
            model_type = env_params['model_type'],
            device = device
        )

        self.in_action_net = ActionNetwork(
            in_dim=env_params['env_dim'],
            hidden_dim=action_params['hidden_dim'],
            model_type=action_params['model_type'],
            num_layers=action_params['num_layers'],
            dropout=action_params['dropout'],
            mlp_func=gumbel_params['gin_mlp_func'],
            device=device
        )

        self.out_action_net = ActionNetwork(
            in_dim=env_params['env_dim'],
            hidden_dim=action_params['hidden_dim'],
            model_type=action_params['model_type'],
            num_layers=action_params['num_layers'],
            dropout=action_params['dropout'],
            mlp_func=gumbel_params['gin_mlp_func'],
            device=device
        )

        # Gumbel参数
        self.learn_temp = gumbel_params['learn_temp']
        self.tau = nn.Parameter(torch.tensor(gumbel_params['tau0']),
                                requires_grad=self.learn_temp)
        self.device = device
        self.num_layers = env_params['num_layers']
        self.dropout = Dropout(env_params['dropout'])
        self.act_func = ReLU()
        self.norm = nn.LayerNorm(env_params['env_dim'])
        self.temp_model = TempSoftPlus(
            learn_temp=gumbel_params['learn_temp'],
            tau0=gumbel_params['tau0'],
            env_dim=env_params['env_dim'],
            gin_mlp_func=gumbel_params['gin_mlp_func'],
            temp_model_type=gumbel_params['model_type'],
            device=device,
            temp=gumbel_params['temp'],
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        result = 0
        x = self.env_net.encoder(x)
        x = self.dropout(x)
        x = self.act_func(x)


        for idx, layer in enumerate(self.env_net.component):
            x = self.norm(x)
            in_logits = self.in_action_net(x, edge_index=edge_index)
            out_logits = self.out_action_net(x, edge_index=edge_index)
            temp = self.temp_model(x=x, edge_index=edge_index)
            in_prob = F.gumbel_softmax(in_logits, hard=True, tau=temp)
            out_prob = F.gumbel_softmax(out_logits, hard=True, tau=temp)
            in_edge = in_prob[:,0]
            out_edge = out_prob[:,0]
            u, v = edge_index
            edge_weight = in_edge[u] * out_edge[v]
            out = self.env_net.component[idx](x=x, edge_index=edge_index, edge_weight=edge_weight)
            out = self.dropout(out)
            out = self.act_func(out)
            x = out
        x = self.norm(x)
        x = self.env_net.decoder(x)
        result = result + x
        return result





