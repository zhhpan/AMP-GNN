
import torch
from torch import nn
from torch.nn import Linear, Dropout, GELU

from layers.load_helper import get_component_list
from lrgb.encoders.composition import Concat2NodeEncoder
from lrgb.encoders.mol_encoder import AtomEncoder


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
                 device: torch.device,
                 is_lrgb: bool):
        super().__init__()

        # 编码器
        if is_lrgb:
            self.encoder = AtomEncoder(env_dim)
        else:
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


