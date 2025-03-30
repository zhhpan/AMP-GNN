import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Dropout, ReLU

from layers.temp import TempSoftPlus
from network.action import ActionNetwork
from network.environment import EnvironmentNetwork
from param import GumbelParameters, EnvironmentParameters, ActionParameters


class AMPGNN(nn.Module):

    def __init__(self,
                 gumbel_params: GumbelParameters,  # 单独参数
                 env_params: EnvironmentParameters,
                 action_params: ActionParameters,
                 device: torch.device):
        super().__init__()

        # 初始化子网络
        self.env_net = EnvironmentNetwork(
            in_dim=env_params.in_dim,
            env_dim=env_params.env_dim,
            out_dim=env_params.out_dim,
            num_layers=env_params.num_layers,
            dropout=env_params.dropout,
            mlp_func=gumbel_params.gin_mlp_func,
            model_type = env_params.model_type,
            device = device
        )

        self.in_action_net = ActionNetwork(
            in_dim=env_params.env_dim,
            hidden_dim=action_params.hidden_dim,
            model_type=action_params.model_type,
            num_layers=action_params.num_layers,
            dropout=action_params.dropout,
            mlp_func=gumbel_params.gin_mlp_func,
            device=device
        )

        self.out_action_net = ActionNetwork(
            in_dim=env_params.env_dim,
            hidden_dim=action_params.hidden_dim,
            model_type=action_params.model_type,
            num_layers=action_params.num_layers,
            dropout=action_params.dropout,
            mlp_func=gumbel_params.gin_mlp_func,
            device=device
        )

        # Gumbel参数
        self.learn_temp = gumbel_params.learn_temp
        self.tau = nn.Parameter(torch.tensor(gumbel_params.tau0),
                                requires_grad=self.learn_temp)
        self.device = device
        self.num_layers = env_params.num_layers
        self.dropout = Dropout(env_params.dropout)
        self.act_func = ReLU()
        self.norm = nn.LayerNorm(env_params.env_dim)
        self.temp_model = TempSoftPlus(
            learn_temp=gumbel_params.learn_temp,
            tau0=gumbel_params.tau0,
            env_dim=env_params.env_dim,
            gin_mlp_func=gumbel_params.gin_mlp_func,
            temp_model_type=gumbel_params.model_type,
            device=device,
            temp=gumbel_params.temp,
        )
        self.layer_probs = {
            'in_prob': [[] for _ in range(self.num_layers)],
            'out_prob': [[] for _ in range(self.num_layers)]
        }
        self.similarity_stats = [[] for _ in range(env_params.num_layers)]

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
            with torch.no_grad():
                self.layer_probs['in_prob'][idx].append(in_prob[:, 0].mean().item())
                self.layer_probs['out_prob'][idx].append(out_prob[:, 0].mean().item())
            in_edge = in_prob[:,0]
            out_edge = out_prob[:,0]
            u, v = edge_index
            edge_weight = in_edge[u] * out_edge[v]
            out = self.env_net.component[idx](x=x, edge_index=edge_index, edge_weight=edge_weight)
            out = self.dropout(out)
            out = self.act_func(out)
            x = out + x
            self._calc_layer_similarity(x,edge_index,idx)
        x = self.norm(x)
        x = self.env_net.decoder(x)
        result = result + x
        return result

    def _init_probs_storage(self):
        """初始化概率存储结构"""
        self.layer_probs = {
            'in_prob': [[] for _ in range(self.num_layers)],
            'out_prob': [[] for _ in range(self.num_layers)]
        }

    def get_probs(self):
        """获取当前概率并清空存储（返回标量列表）"""
        probs = {
            'in_prob': [np.mean(layer) if layer else 0 for layer in self.layer_probs['in_prob']],
            'out_prob': [np.mean(layer) if layer else 0 for layer in self.layer_probs['out_prob']]
        }
        # 清空存储
        self._init_probs_storage()
        return probs

    def _calc_layer_similarity(self, features: Tensor, edge_index, layer_idx: int):
        """计算余弦相似度（保持向量化计算）"""
        src, dst = edge_index
        # 向量化计算特征差异
        diff = features[src] - features[dst]
        squared_diff = diff.pow(2).sum(dim=1)
        num_edges = edge_index.shape[1]
        num_features = features.shape[1]
        self.similarity_stats[layer_idx].append((squared_diff.sum() / (num_features * num_edges)).item())