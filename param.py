from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class GumbelParameters:
    """Gumbel Softmax相关参数配置"""
    learn_temp: bool
    tau0: float
    temp: float
    gin_mlp_func: Callable[[], Any]
    model_type: str


@dataclass
class EnvironmentParameters:
    """环境网络核心参数配置"""
    num_layers: int
    env_dim: int
    in_dim: int
    out_dim: int
    dropout: float
    activation: Callable[[], Any]
    model_type: str


@dataclass
class ActionParameters:
    """行动网络参数配置"""
    num_layers: int
    hidden_dim: int
    env_dim: int
    dropout: float
    model_type: str




