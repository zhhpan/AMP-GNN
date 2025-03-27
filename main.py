import logging
import os
import os.path as osp
from argparse import Namespace
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataset import DataSet as DS
from model import AMPGNN
from param import GumbelParameters, EnvironmentParameters, ActionParameters


class Experiment:

    def __init__(self, args: Namespace) -> None:
        """
        实验配置初始化
        Args:
            args (Namespace): 包含以下关键参数：
                - dataset_name: 数据集名称
                - seed: 随机种子
                - batch_size: 批大小
                - env_dim: 环境网络维度
                - act_dim: 动作网络维度
                - dropout: 丢弃率
                - lr: 学习率
                - epochs: 训练轮次

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

        # 获取分类任务类别数
        self.num_classes = self.dataset.get_out_dim()
        self.logger.info(f"分类类别数: {self.num_classes}")

        # 初始化Accuracy指标
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=self.num_classes
        ).to(self.device)

    def set_seed(self) -> None:
        """设置全局随机种子"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        self.logger.info(f"全局随机种子设置为 {self.seed}")

    def prepare_model_arguments(self) :
        """

        Returns:
            'gumbel': Gumbel参数
            'env': 环境网络参数
            'action': 行动网络参数
        """

        # Gumbel参数
        gumbel_params = GumbelParameters(
            learn_temp = self.learn_temp,
            tau0 = self.tau0,
            temp = self.temp,
            gin_mlp_func = self.dataset.gin_mlp_func(),
            model_type = self.gumbel_model_type
        )

        # 环境网络参数（核心组件）
        env_params = EnvironmentParameters(
            num_layers = self.env_num_layers,
            env_dim = self.env_dim,
            in_dim = self.dataset.num_features,
            out_dim = self.dataset.get_out_dim(),
            dropout = self.dropout,
            activation = self.dataset.env_activation_type(),
            model_type = self.env_model_type
        )

        # 行动网络参数
        action_params = ActionParameters(
            num_layers = self.act_num_layers,
            hidden_dim = self.act_dim,
            env_dim = self.env_dim,
            dropout = self.dropout,
            model_type = self.act_model_type
        )

        return gumbel_params, env_params, action_params

    def load_dataset(self) :
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

        return

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
                # correct += pred.eq(data.y.to(self.device)).sum().item()
                node_mask = getattr(data, f"{mask}_mask")
                self.accuracy(pred, data.y.to(self.device))

        avg_loss = total_loss / len(loader)
        #accuracy = correct / len(loader.dataset[0].y)
        accuracy = self.accuracy.compute().item()
        return avg_loss, accuracy

    def run(self) -> Dict[str, Any]:
        """执行10折交叉验证流程（集成动态损失图）"""
        fold_results = []
        total_steps = len(self.folds) * self.epochs

        # 创建主进度条
        with tqdm(total=total_steps, desc="🌐 初始化训练进度...") as pbar:
            for fold in range(len(self.folds)):
                # 初始化折叠相关变量
                data_fold = self.dataset.select_fold_and_split(fold)
                self.data = data_fold
                loaders = self.create_data_loaders()

                # 初始化模型
                gumbel_params, env_params, action_params = self.prepare_model_arguments()
                model = AMPGNN(gumbel_params, env_params, action_params, self.device).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)

                # 初始化本折叠的损失记录
                train_losses = []
                val_losses = []
                best_acc = 0

                # 创建动态图表
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name='Train Loss',
                    line=dict(color='blue')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name='Val Loss',
                    line=dict(color='red')
                ), row=1, col=1)
                fig.update_layout(
                    title=f'Fold {fold + 1} Training Progress',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    showlegend=True,
                    template='plotly_white'
                )

                # 初始化HTML文件路径
                html_path = f'result/training_fold_{fold+1}.html'
                fig.write_html(html_path, auto_open=False)

                for epoch in range(self.epochs):
                    # 训练步骤
                    model.train()
                    optimizer.zero_grad()
                    train_loss = self.calculate_loss(model, self.data, 'train')
                    train_loss.backward()
                    optimizer.step()

                    # 验证步骤
                    model.eval()
                    with torch.no_grad():
                        val_loss, val_acc = self.evaluate(model, loaders['val'], 'val')
                    self.accuracy.reset()

                    # 记录损失
                    train_losses.append(train_loss.item())
                    val_losses.append(val_loss)

                    # 动态更新图表（每50个epoch更新一次）
                    if epoch % 50 == 0 or epoch == self.epochs - 1:
                        # 更新图表数据
                        fig.update_traces(
                            x=np.arange(len(train_losses)),
                            y=train_losses,
                            selector={'name': 'Train Loss'}
                        )
                        fig.update_traces(
                            x=np.arange(len(val_losses)),
                            y=val_losses,
                            selector={'name': 'Val Loss'}
                        )

                        # 自动调整坐标轴范围
                        fig.update_xaxes(range=[0, self.epochs])
                        y_max = max(max(train_losses), max(val_losses)) * 1.1
                        fig.update_yaxes(range=[0, y_max])

                        # 保存更新后的图表
                        fig.write_html(html_path, auto_open=False)

                    # 更新最佳模型
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), f'fold/{self.dataset.name}_fold{fold}.pth')

                    # 更新进度条
                    desc = (f"\033[32m🌐 Fold {fold + 1}/{len(self.folds)} | "
                            f"\033[34mEpoch {epoch + 1}/{self.epochs} | "
                            f"Train: {train_loss.item():.4f} | "
                            f"Val: {val_loss:.4f} | "
                            f"Best Val Acc: {best_acc:.2%}")
                    pbar.set_description(desc)
                    pbar.update(1)

                # 折叠结束后显示最终结果
                model.load_state_dict(torch.load(f'fold/{self.dataset.name}_fold{fold}.pth'))
                test_loss, test_acc = self.evaluate(model, loaders['test'], 'test')
                fold_results.append(test_acc)

                # 添加最终标注
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.95, y=0.95,
                    text=f"Final Test Acc: {test_acc:.2%}",
                    showarrow=False,
                    font=dict(size=12)
                )
                fig.write_html(html_path, auto_open=False)
                print(f"\n✅ Fold {fold+1} 训练完成，图表已保存至 {html_path}")

        # 生成最终统计图表
        final_fig = go.Figure()
        final_fig.add_trace(go.Bar(
            x=[f'Fold {i+1}' for i in range(len(fold_results))],
            y=fold_results,
            marker_color='rgb(55, 83, 109)'
        ))
        final_fig.update_layout(
            title='10-Fold Cross Validation Results',
            xaxis_title='Fold',
            yaxis_title='Accuracy',
            yaxis_tickformat=".2%",
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    text=f"Mean Accuracy: {np.mean(fold_results):.2%} ± {np.std(fold_results):.2%}",
                    xref="paper",
                    yref="paper"
                )
            ]
        )
        final_fig.write_html('result/final_results.html', auto_open=True)

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
        epochs=10000,
        learn_temp=True,
        tau0 = 0.5,
        temp = 0.01,
        env_num_layers = 6,
        act_num_layers = 1 ,
        env_model_type = 'MEAN_GNN',
        act_model_type = 'MEAN_GNN',
        gumbel_model_type = 'LIN'
    )

    # 运行实验
    experiment = Experiment(args)
    results = experiment.run()

    print(f"10折交叉验证结果:")
    print(f"平均准确率: {results['mean_acc']:.2%} ± {results['std_acc']:.2%}")
    print(f"最佳折叠: {results['max_acc']:.2%}")
    print(f"最差折叠: {results['min_acc']:.2%}")