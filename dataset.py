from typing import Callable, Optional, List
import torch
from torch import nn
from torch_geometric.data import Data
import copy
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset


class DataSet:

    def __init__(self, name: str, root: str, transform: Optional[Callable] = None):
        """
        初始化数据集实例，自动加载数据。

        Args:
            name (str): 数据集名称
            root (str): 数据存储根目录
            transform (Callable, optional): 数据预处理函数
        """
        self.name = name
        self.root = root
        self.transform = T.ToUndirected()

        # 加载数据
        self.data = self.load_data(transform=transform)

    def load_data(self, transform: Optional[Callable] = None) -> Data:
        """
        加载roman_empire数据集，并返回torch_geometric.data.Data对象。

        Args:
            transform (callable, optional): 用于数据转换的函数。

        Returns:
            Data: 加载并处理后的torch_geometric.data.Data对象。
        """
        # 创建HeterophilousGraphDataset对象，只处理roman_empire数据集
        dataset = HeterophilousGraphDataset(root=self.root, name=self.name, transform=T.ToUndirected())
        return dataset[0]  # 返回第一个Data对象


    def select_fold_and_split(self, fold: int):
        """
        根据数据集的掩码返回训练集、验证集和测试集。
        使用节点掩码（train_mask、val_mask、test_mask）来划分数据。
        在10折交叉验证中，返回对应折叠的训练、验证、测试集。

        Args:
            fold (int): 当前折叠索引。

        Returns:
            dataset_copy: Data集合
        """
        dataset_copy = copy.deepcopy(self.data)
        dataset_copy.train_mask = self.data.train_mask[:, fold]
        dataset_copy.val_mask = self.data.val_mask[:, fold]
        dataset_copy.test_mask = self.data.test_mask[:, fold]

        return dataset_copy



    def get_folds(self):
        """
        返回10折交叉验证的折叠索引。
        每个折叠索引（0到9）表示一个训练和验证集划分。

        Returns:
            list: 包含每个折叠的索引列表，例如 [0, 1, 2, ..., 9]。
        """
        return list(range(10))  # 返回0到9的折叠索引


    def get_out_dim(self) -> int:
        """
        对于roman_empire数据集，确定模型输出层的维度。

        Returns:
            int: 输出层维度。
        """
        return int(max([self.data.y.max().item() ]) + 1)

    # 7. GIN网络的MLP构建函数
    def gin_mlp_func(self) -> Callable:
        """
        针对roman_empire数据集，构建GIN网络的等宽MLP结构（无中间层扩展和BatchNorm）。

        Returns:
            Callable: MLP构建函数，输入参数为(in_channels, out_channels, bias)。
        """
        def mlp_func(in_channels: int, out_channels: int, bias: bool):
            return torch.nn.Sequential(
                torch.nn.Linear(in_channels, out_channels, bias=bias),  # 输入层
                torch.nn.ReLU(),                                         # 激活函数
                torch.nn.Linear(out_channels, out_channels, bias=bias)  # 输出层（等宽）
            )
        return mlp_func

    @property
    def num_features(self) -> int:
        """获取特征维度（适配roman-empire的300维）"""
        return self.data.x.shape[1]
        # temp = self.data.x[0,:]
        # return temp.shape

    def env_activation_type(self) -> type:
        """GIN的激活函数类型"""
        return nn.ReLU
