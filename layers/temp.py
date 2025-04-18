from typing import Callable

from torch import nn, Tensor
from torch.cuda import device
from torch.nn import Module, ModuleList
from torch_geometric.typing import Adj

from layers.load_helper import get_component_list


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

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        x = self.linear_model[0](x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.softplus(x) + self.tau0
        temp = x.pow_(-1)
        return temp.masked_fill_(temp == float('inf'), 0.)