import os
from argparse import Namespace
from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import logging
import os.path as osp

from tqdm import tqdm

from dataset import DataSet as DS
from model import CoGNN


class Experiment:

    def __init__(self, args: Namespace) -> None:
        """
        实验配置初始化
        Args:
            args (Namespace): 包含以下关键参数：
                - dataset_name: 数据集名称
                - seed: 随机种子（默认42）
                - batch_size: 批大小
                - env_dim: 环境网络维度（默认64）
                - act_dim: 行动网络维度（默认32）
                - dropout: 丢弃率（默认0.5）
                - lr: 学习率（默认0.001）
                - epochs: 训练轮次（默认3000）

        """
        # 初始化日志配置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
        self.logger = logging.getLogger(__name__)

        # 参数注入
        for param in vars(args):
            value = getattr(args, param)
            self.logger.info(f"初始化参数 {param}: {value}")
            setattr(self, param, value)

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"运行设备: {self.device}")

        # 初始化随机种子
        self.set_seed()

        # 初始化数据集
        self.load_dataset()

        # 初始化任务损失函数
        self.task_loss = CrossEntropyLoss()

        # 获取所有折叠索引
        self.folds = self.dataset.get_folds()

    def set_seed(self) -> None:
        """设置全局随机种子"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        self.logger.info(f"全局随机种子设置为 {self.seed}")

    def prepare_model_arguments(self) -> Dict[str, Dict]:
        """

        Returns:
            Dict: 包含三个参数字典：
                'gumbel': Gumbel参数
                'env': 环境网络参数
                'action': 行动网络参数
        """
        # Gumbel参数
        gumbel_params = {
            'learn_temp': self.learn_temp,
            'tau0': self.tau0,
            'temp': self.temp,
            'gin_mlp_func': self.dataset.gin_mlp_func(),
            'model_type': self.env_model_type
        }

        # 环境网络参数（核心组件）
        env_params = {
            'num_layers': self.env_num_layers,
            'env_dim': self.env_dim,
            'in_dim': self.dataset.num_features,
            'out_dim': self.dataset.get_out_dim(),
            'dropout': self.dropout,
            'activation': self.dataset.env_activation_type(),
            'model_type' : self.env_model_type
        }

        # 行动网络参数
        action_params = {
            'num_layers': self.act_num_layers,
            'hidden_dim': self.act_dim,
            'env_dim': self.env_dim,
            'dropout': self.dropout,
            'model_type': self.act_model_type
        }

        return gumbel_params, env_params, action_params

    def load_dataset(self) -> Data:
        """

        Returns:
            Data: 包含以下属性的数据对象：
                - x: 节点特征
                - y: 节点标签
                - edge_index: 全连接边索引
                - train_mask: 训练节点掩码
                - val_mask: 验证节点掩码
                - test_mask: 测试节点掩码
        """
        # 初始化dataset的参数和存储位置
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        root = osp.join(ROOT_DIR, 'datasets')
        name = self.dataset_name
        # 加载数据集
        self.dataset = DS(root=root, name=name)
        # 类型转换
        self.dataset.data.y = self.dataset.data.y.to(torch.long)

        return self.dataset.data

    def create_data_loaders(self) -> Dict[str, DataLoader]:
        """
            创建数据加载器

            Returns:
                Dict: 包含三个数据加载器：
                    'train': 训练集加载器
                    'val': 验证集加载器
                    'test': 测试集加载器
         """
        def apply_mask(data: Data, mask_type: str) -> Data:
            masked_data = data.clone()
            for node_type in ['train', 'val', 'test']:
                if node_type != mask_type:
                    setattr(masked_data, f"{node_type}_mask", None)
            return masked_data

        loaders = {
            'train': DataLoader([apply_mask(self.data, 'train')],
                                batch_size=self.batch_size, shuffle=False),
            'val': DataLoader([apply_mask(self.data, 'val')],
                              batch_size=self.batch_size, shuffle=False),
            'test': DataLoader([apply_mask(self.data, 'test')],
                               batch_size=self.batch_size, shuffle=False)
        }
        return loaders

    def calculate_loss(self, model: torch.nn.Module, data: Data, mask: str) -> Tensor:
        """
        计算交叉熵损失

        Args:
            model: 训练中的模型
            data: 当前批次数据
            mask: 使用的掩码名称（train/val/test）

        Returns:
            Tensor: 计算得到的损失值
        """
        # 获取节点掩码
        node_mask = getattr(data, f"{mask}_mask")

        # 前向传播
        out = model(data.x.to(self.device),
                    data.edge_index.to(self.device))

        # 计算损失
        loss = self.task_loss(out[node_mask], data.y.to(self.device)[node_mask])
        return loss

    def evaluate(self, model: torch.nn.Module, loader: DataLoader, mask: str) -> Tuple[float, float]:
        """模型评估"""
        model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for data in loader:
                loss = self.calculate_loss(model, data, mask)
                total_loss += loss.item()

                pred = model(data.x.to(self.device),
                             data.edge_index.to(self.device)).argmax(dim=1)
                correct += pred.eq(data.y.to(self.device)).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = correct / len(loader.dataset[0].y)
        return avg_loss, accuracy

    def run(self) -> Dict[str, Any]:
        """执行10折交叉验证流程"""
        fold_results = []
        total_steps = len(self.folds) * self.epochs

        # 创建主进度条（总步数=折叠数×epoch数）
        with tqdm(
                total=total_steps,
                desc="🌐 初始化训练进度...",  # 初始描述
                bar_format="{desc}  [已用:{elapsed} 剩余:{remaining}]",
                mininterval=0.5  # 降低刷新频率
        ) as pbar:
            print("")
            for fold in range(len(self.folds)):
                # 初始化当前折叠
                data_fold = self.dataset.select_fold_and_split(fold)
                self.data = data_fold
                loaders = self.create_data_loaders()

                # 初始化模型
                gumbel_params, env_params, action_params = self.prepare_model_arguments()
                model = CoGNN(gumbel_params, env_params, action_params, self.device).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)

                best_acc = 0
                for epoch in range(self.epochs):
                    # 训练步骤
                    model.train()
                    optimizer.zero_grad()
                    loss = self.calculate_loss(model, self.data, 'train')
                    loss.backward()
                    optimizer.step()

                    # 验证步骤（每10个epoch验证一次）
                    if epoch % 10 == 0 or epoch == self.epochs - 1:
                        model.eval()
                        with torch.no_grad():
                            _, val_acc = self.evaluate(model, loaders['val'], 'val')

                        # 更新最佳模型
                        if val_acc > best_acc:
                            best_acc = val_acc
                            torch.save(model.state_dict(), f'fold/{self.dataset.name}_fold{fold}.pth')

                    # 动态更新进度条描述
                    desc = (
                        f"\033[32m🌐 折叠Fold {fold + 1}/{len(self.folds)}\033[0m | "
                        f"\033[34m轮次Epoch {epoch + 1}/{self.epochs}\033[0m | "
                        f"损失: \033[31m{loss.item():.4f}\033[0m | "
                        f"最佳验证: \033[33m{best_acc:.2%}\033[0m"
                    )
                    pbar.set_description(desc)
                    pbar.update(1)

                # 折叠训练完成，执行测试
                model.load_state_dict(torch.load(f'fold/{self.dataset.name}_fold{fold}.pth'))
                _, test_acc = self.evaluate(model, loaders['test'], 'test')
                fold_results.append(test_acc)

                # 更新最终结果展示
                pbar.write(
                    f"\n✅ Fold {fold + 1} Completed | "
                    f"Test Accuracy: {test_acc:.2%} | "
                    f"Current Mean: {np.mean(fold_results):.2%}"
                )

        # 统计结果
        test_accs = torch.tensor(fold_results)
        return {
            'fold_accs': test_accs.tolist(),
            'mean_acc': test_accs.mean().item(),
            'std_acc': test_accs.std().item(),
            'max_acc': test_accs.max().item(),
            'min_acc': test_accs.min().item()
        }


# 示例用法
if __name__ == "__main__":
    # 初始化参数
    args = Namespace(
        dataset_name='roman-empire',
        seed=0,
        batch_size=32,
        env_dim=64,
        act_dim=16,
        dropout=0.2,
        lr=0.001,
        epochs=3000,
        learn_temp=True,
        tau0 = 0.5,
        temp = 0.01,
        env_num_layers = 3,
        act_num_layers = 1 ,
        env_model_type = 'GIN',
        act_model_type = 'GCN',
    )

    # 运行实验
    experiment = Experiment(args)
    results = experiment.run()

    print(f"10折交叉验证结果:")
    print(f"平均准确率: {results['mean_acc']:.2%} ± {results['std_acc']:.2%}")
    print(f"最佳折叠: {results['max_acc']:.2%}")
    print(f"最差折叠: {results['min_acc']:.2%}")