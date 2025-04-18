import copy
from typing import Callable, Optional

import numpy as np
import torch
import torch_geometric.transforms as T
from torch import nn
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
from torch_geometric.nn import Node2Vec

from lrgb.lrgb import PeptidesFunctionalDataset
from lrgb.transforms import apply_transform


class DataSet:

    def __init__(self, name: str, root: str):
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
        self.data = self.load_data()
        # self.generate_node2vec_embedding(self.num_features)

    def load_data(self) -> Data:
        """
        加载数据集，并返回torch_geometric.data.Data对象。

        Returns:
            Data: 加载并处理后的torch_geometric.data.Data对象。
        """
        if self.name in ['roman_empire', 'amazon_ratings', 'minesweeper',
                    'tolokers', 'questions']:
            # 创建HeterophilousGraphDataset对象
            dataset = HeterophilousGraphDataset(root=self.root, name=self.name, transform=T.ToUndirected())
            return dataset[0]  # 返回第一个Data对象
        elif self.name in['cora', 'pubmed']:
            dataset = Planetoid(root=self.root, name=self.name, transform=T.NormalizeFeatures())
            return dataset[0]  # 返回第一个Data对象
        elif self.name in ['func']:
            dataset = PeptidesFunctionalDataset(root=self.root)
            pos_enc = 16
            dataset = apply_transform(dataset=dataset, pos_encoder=pos_enc)
            return dataset[0]
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')


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
        if self.name in ['roman_empire', 'amazon_ratings', 'minesweeper',
                         'tolokers', 'questions']:
            dataset_copy = copy.deepcopy(self.data)
            dataset_copy.train_mask = self.data.train_mask[:, fold]
            dataset_copy.val_mask = self.data.val_mask[:, fold]
            dataset_copy.test_mask = self.data.test_mask[:, fold]
            return dataset_copy
        elif self.name in ['cora', 'pubmed']:
            device = self.data.x.device
            with np.load(f'folds/{self.name}_split_0.6_0.2_{fold}.npz') as folds_file:
                train_mask = torch.tensor(folds_file['train_mask'], dtype=torch.bool, device=device)
                val_mask = torch.tensor(folds_file['val_mask'], dtype=torch.bool, device=device)
                test_mask = torch.tensor(folds_file['test_mask'], dtype=torch.bool, device=device)

            setattr(self.data, 'train_mask', train_mask)
            setattr(self.data, 'val_mask', val_mask)
            setattr(self.data, 'test_mask', test_mask)

            # self.data.train_mask[self.data.non_valid_samples] = False
            # self.data.test_mask[self.data.non_valid_samples] = False
            # self.data.val_mask[self.data.non_valid_samples] = False
            return self.data
        elif self.name in ['func']:
            split_idx = self.data.get_idx_split()
            train_indices = split_idx["train"]
            val_indices = split_idx["val"]
            test_indices = split_idx["test"]
            train_mask = Subset(self.data, train_indices)
            val_mask = Subset(self.data, val_indices)
            test_mask = Subset(self.data, test_indices)
            setattr(self.data, 'train_mask', train_mask)
            setattr(self.data, 'val_mask', val_mask)
            setattr(self.data, 'test_mask', test_mask)
            return self.data


    def get_folds(self):
        """
        返回10折交叉验证的折叠索引。
        每个折叠索引（0到9）表示一个训练和验证集划分。

        Returns:
            list: 包含每个折叠的索引列表，例如 [0, 1, 2, ..., 9]。
        """
        if self.name in ['func']:
            return list(range(1))
        else:
            return list(range(10))  # 返回0到9的折叠索引


    def get_out_dim(self) -> int:
        """
        确定模型输出层的维度。

        Returns:
            int: 输出层维度。
        """
        if self.name in ['func']:
            return self.data.y.shape[1]
        else:
            return int(max([self.data.y.max().item() ]) + 1)

    # 7. GIN网络的MLP构建函数
    def gin_mlp_func(self) -> Callable:
        """
        构建GIN网络的等宽MLP结构（无中间层扩展和BatchNorm）。

        Returns:
            Callable: MLP构建函数，输入参数为(in_channels, out_channels, bias)。
        """
        if self.name == 'func':
            def mlp_func(in_channels: int, out_channels: int, bias: bool):
                return torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels, bias=bias),
                                           torch.nn.ReLU(), torch.nn.Linear(out_channels, out_channels, bias=bias))
        else:
            def mlp_func(in_channels: int, out_channels: int, bias: bool):
                return torch.nn.Sequential(torch.nn.Linear(in_channels, 2 * in_channels, bias=bias),
                                               torch.nn.BatchNorm1d(2 * in_channels),
                                               torch.nn.ReLU(), torch.nn.Linear(2 * in_channels, out_channels, bias=bias))
        return mlp_func

    @property
    def num_features(self) -> int:
        """获取特征维度"""
        return self.data.x.shape[1]


    def env_activation_type(self) -> type:
        """GIN的激活函数类型"""
        return nn.ReLU

    def generate_node2vec_embedding(self, embedding_dim, walk_length=30, context_size=10):
        """
        生成Node2Vec嵌入并与原始特征拼接
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化Node2Vec模型
        self.node2vec_model = Node2Vec(
            edge_index=self.data.edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            sparse=True
        ).to(device)

        # 训练嵌入模型
        loader = self.node2vec_model.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(self.node2vec_model.parameters(), lr=0.01)

        self.node2vec_model.train()
        for epoch in range(100):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.node2vec_model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

        # 合并原始特征与嵌入
        self.data.x = self._combine_features(embedding_dim)

    def _combine_features(self, embedding_dim):
        """将原始特征与Node2Vec嵌入拼接"""
        node_ids = torch.arange(self.data.num_nodes, dtype=torch.long)
        embeddings = self.node2vec_model(node_ids.to(self.node2vec_model.embedding.weight.device)).detach()

        # 标准化处理
        embeddings = (embeddings - embeddings.mean(dim=0)) / embeddings.std(dim=0)
        original_features = self.data.x.to(embeddings.device)

        return torch.cat([original_features, embeddings], dim=1)