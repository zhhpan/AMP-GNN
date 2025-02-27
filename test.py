import argparse
import logging
import os
import unittest

from torch import nn
from torch.nn import Linear, GELU, Dropout

from dataset import DataSet as DS
import os.path as osp

class MyTestCase(unittest.TestCase):
    def setUp(self):
        """初始化日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def test_dataset(self):
        # 设置根目录路径
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        root = osp.join(ROOT_DIR, 'datasets')
        name = 'roman-empire'

        # 加载数据集
        dataset = DS(root=root, name=name)
        print('dataset:', dataset.data)
        # 检查 root 目录结构


        # 拆分数据集为训练集、验证集和测试集
        folds = dataset.get_folds()
        # 获取模型输出维度
        out_dim = dataset.get_out_dim()
        print('out_dim:', out_dim)
        # GIN网络的MLP构建函数
        gin_mlp_func = dataset.gin_mlp_func()

        for fold in folds:
            dataset_by_split = dataset.select_fold_and_split(fold=fold)
            # # 打印节点特征的形状
            # print(dataset_by_split.train_mask)
            # print(dataset_by_split.val_mask)
            # print(dataset_by_split.test_mask)
            # print("---------------------------------------------")

    def test_arg(self):
        args = argparse.Namespace(
            dataset_name='roman-empire',
            seed=42,
            batch_size=1,
            env_dim=64,
            act_dim=32,
            dropout=0.5,
            lr=0.001,
            epochs=200
        )
        logger = logging.getLogger(__name__)
        # 参数注入
        for param in vars(args):
            value = getattr(args, param)
            logger.info(f"初始化参数 {param}: {value}")
            setattr(self, param, value)
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        root = osp.join(ROOT_DIR, 'datasets')
        name = self.dataset_name

        # 加载数据集
        dataset = DS(root, name)
        print('dataset:', dataset.data)

    def test_model_list(self):
        self.convs = nn.ModuleList()
        # 编码器 使用线性层
        self.convs.append(Linear(2, 3))

        # 解码器
        self.convs.extend([nn.Linear(5, 4), nn.Dropout(p=0.5), nn.GELU()])  # 正确写法

        print('convs:', self.convs)


if __name__ == '__main__':
    unittest.main()
