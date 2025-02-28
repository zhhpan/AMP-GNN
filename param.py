from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class GumbelParameters:
    """Gumbel Softmax相关参数配置"""
    learn_temp: bool
    tau0: float
    temp: float
    gin_mlp_func: Callable[[], Any]  # 生成GIN网络的MLP函数
    model_type: str


@dataclass
class EnvironmentParameters:
    """环境网络核心参数配置"""
    num_layers: int
    env_dim: int
    in_dim: int  # 输入维度来自数据集
    out_dim: int  # 输出维度来自数据集
    dropout: float
    activation: Callable[[], Any]  # 激活函数类型
    model_type: str


@dataclass
class ActionParameters:
    """行动网络参数配置"""
    num_layers: int
    hidden_dim: int
    env_dim: int  # 与环境网络共享维度
    dropout: float
    model_type: str




