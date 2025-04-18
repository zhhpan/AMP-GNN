import logging
import os
import os.path as osp
from argparse import Namespace
from typing import Any, Dict, Tuple, List
import threading
import time

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

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# 假设这些模块已正确实现
from dataset import DataSet as DS
from lrgb.encoders.mol_encoder import BondEncoder
from model import AMPGNN
from param import GumbelParameters, EnvironmentParameters, ActionParameters

# 全局状态变量
global_training_progress = {}
global_final_results = {}

class Experiment:
    def __init__(self, args: Namespace) -> None:
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

        # 早停参数
        self.early_stop_patience = getattr(args, 'early_stop_patience', 100)
        self.early_stop_delta = getattr(args, 'early_stop_delta', 0.001)
        self.logger.info(f"早停参数 - patience: {self.early_stop_patience}, delta: {self.early_stop_delta}")

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"运行设备: {self.device}")

        # 初始化随机种子
        self.set_seed()

        # 初始化数据集
        self.load_dataset()

        # 损失函数
        self.task_loss = CrossEntropyLoss()

        # 折叠信息
        self.folds = self.dataset.get_folds()
        self.num_classes = self.dataset.get_out_dim()
        self.logger.info(f"分类类别数: {self.num_classes}")

        # 评估指标
        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=self.num_classes
        ).to(self.device)

    def set_seed(self) -> None:
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        self.logger.info(f"全局随机种子设置为 {self.seed}")

    def prepare_model_arguments(self):
        gumbel_params = GumbelParameters(
            learn_temp=self.learn_temp,
            tau0=self.tau0,
            temp=self.temp,
            gin_mlp_func=self.dataset.gin_mlp_func(),
            model_type=self.gumbel_model_type
        )
        env_params = EnvironmentParameters(
            num_layers=self.env_num_layers,
            env_dim=self.env_dim,
            in_dim=self.dataset.num_features,
            out_dim=self.dataset.get_out_dim(),
            dropout=self.dropout,
            activation=self.dataset.env_activation_type(),
            model_type=self.env_model_type,
            is_lrgb = True if self.dataset_name == 'func' else False
        )
        action_params = ActionParameters(
            num_layers=self.act_num_layers,
            hidden_dim=self.act_dim,
            env_dim=self.env_dim,
            dropout=self.dropout,
            model_type=self.act_model_type
        )
        return gumbel_params, env_params, action_params

    def load_dataset(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        root = osp.join(ROOT_DIR, 'datasets')
        name = self.dataset_name
        self.dataset = DS(root=root, name=name)
        self.dataset.data.y = self.dataset.data.y.to(torch.long)

    def create_data_loaders(self) -> Dict[str, DataLoader]:
        def apply_mask(data: Data, mask_type: str) -> Data:
            masked_data = data.clone()
            for node_type in ['train', 'val', 'test']:
                if node_type != mask_type:
                    setattr(masked_data, f"{node_type}_mask", None)
            return masked_data

        return {
            'train': DataLoader([apply_mask(self.data, 'train')], batch_size=self.batch_size, shuffle=False),
            'val': DataLoader([apply_mask(self.data, 'val')], batch_size=self.batch_size, shuffle=False),
            'test': DataLoader([apply_mask(self.data, 'test')], batch_size=self.batch_size, shuffle=False)
        }

    def calculate_loss(self, model: torch.nn.Module, data: Data, mask: str) -> Tensor:
        node_mask = getattr(data, f"{mask}_mask")
        out = model(x=  data.x.to(self.device), data.edge_index.to(self.device))
        return self.task_loss(out[node_mask], data.y.to(self.device)[node_mask])

    def evaluate(self, model: torch.nn.Module, loader: DataLoader, mask: str) -> Tuple[float, float]:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                loss = self.calculate_loss(model, data, mask)
                total_loss += loss.item()
                pred = model(data.x.to(self.device), data.edge_index.to(self.device)).argmax(dim=1)
                node_mask = getattr(data, f"{mask}_mask")
                self.accuracy(pred, data.y.to(self.device))
        avg_loss = total_loss / len(loader)
        accuracy = self.accuracy.compute().item()
        self.accuracy.reset()
        return avg_loss, accuracy

    def run(self) -> Dict[str, Any]:
        global global_training_progress, global_final_results
        fold_results = []
        total_steps = len(self.folds) * self.epochs
        pbar = tqdm(total=total_steps, desc="🌐 初始化训练进度...")

        for fold_idx in range(len(self.folds)):
            data_fold = self.dataset.select_fold_and_split(fold_idx)
            self.data = data_fold
            loaders = self.create_data_loaders()

            gumbel_params, env_params, action_params = self.prepare_model_arguments()
            if(self.dataset_name == 'func'):
                env_edge_embedding = BondEncoder(env_params.env_dim)
                act_edge_embedding = BondEncoder(action_params.hidden_dim)
            model = AMPGNN(gumbel_params, env_params, action_params, self.device, self.use_model,env_edge_embedding,act_edge_embedding).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)

            train_losses = []
            val_losses = []
            best_acc = 0
            early_stop_counter = 0
            best_epoch = 0

            global_training_progress[fold_idx] = {"epoch": [], "train_loss": [], "val_loss": [],  "layer_probs": [] }

            for epoch in range(self.epochs):
                # 训练步骤
                model.train()
                optimizer.zero_grad()
                train_loss = self.calculate_loss(model, self.data, 'train')
                train_loss.backward()
                optimizer.step()
                layer_probs = model.get_probs()
                global_training_progress[fold_idx]["layer_probs"].append({
                    'in_probs': layer_probs['in_prob'],
                    'out_probs': layer_probs['out_prob']
                })
                # 收集相似度数据
                layer_sims = [np.mean(layer) for layer in model.similarity_stats]
                global_training_progress[fold_idx].setdefault("similarity", []).append(layer_sims)
                model.similarity_stats = [[] for _ in range(len(model.similarity_stats))]  # 清空缓存
                # 验证步骤
                val_loss, val_acc = self.evaluate(model, loaders['val'], 'val')

                # 早停逻辑
                if val_acc > best_acc + self.early_stop_delta:
                    best_acc = val_acc
                    early_stop_counter = 0
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'fold/{self.dataset.name}_fold{fold_idx}.pth')
                else:
                    early_stop_counter += 1

                # 记录训练进度
                train_losses.append(train_loss.item())
                val_losses.append(val_loss)
                global_training_progress[fold_idx]["epoch"].append(epoch)
                global_training_progress[fold_idx]["train_loss"].append(train_loss.item())
                global_training_progress[fold_idx]["val_loss"].append(val_loss)

                # 更新进度
                desc = (f"Fold {fold_idx+1}/{len(self.folds)} | Epoch {epoch+1}/{self.epochs} | "
                       f"Train: {train_loss.item():.4f} | Val: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
                pbar.set_description(desc)
                pbar.update(1)

                # 触发早停
                if early_stop_counter >= self.early_stop_patience:
                    self.logger.info(f"Fold {fold_idx+1} 早停触发于epoch {epoch+1}，最佳epoch {best_epoch+1}")
                    break

                time.sleep(0.001)  # 用于演示的短暂暂停

            # 最终测试
            model.load_state_dict(torch.load(f'fold/{self.dataset.name}_fold{fold_idx}.pth'))
            test_loss, test_acc = self.evaluate(model, loaders['test'], 'test')
            fold_results.append(test_acc)
            self.logger.info(f"Fold {fold_idx+1} 完成，Test Acc: {test_acc:.2%}")

        pbar.close()

        # 结果统计
        test_accs = torch.tensor(fold_results)
        global_final_results = {
            "folds": test_accs.tolist(),
            "mean": test_accs.mean().item(),
            "std": test_accs.std().item(),
            "max": test_accs.max().item(),
            "min": test_accs.min().item()
        }
        return global_final_results

# Dash应用
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("GNN 训练实时监控"),
    html.Div(id='training-status', children="等待训练数据更新..."),
    dcc.Dropdown(
        id='fold-selector',
        options=[{'label': f'Fold {i+1}', 'value': i} for i in range(10)],
        value=0,
        clearable=False
    ),
    dcc.Graph(id='progress-graph'),
    html.Hr(),
    html.H2("10折交叉验证结果"),
    dcc.Graph(id='final-results-graph'),
    dcc.Interval(id='interval-component', interval=3000, n_intervals=0),
    html.Hr(),
    html.H2("层概率监控"),
    dcc.Graph(id='layer-probs-graph'),
    dcc.Interval(id='interval-component', interval=3000, n_intervals=0),
    html.Hr(),
    html.H2("分层相似度动态"),
    dcc.Graph(id='similarity-graph'),
])

@app.callback(
    Output('progress-graph', 'figure'),
    Output('training-status', 'children'),
    Input('fold-selector', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_progress(selected_fold, n_intervals):
    if selected_fold in global_training_progress and global_training_progress[selected_fold]["epoch"]:
        fold_data = global_training_progress[selected_fold]
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(
            x=fold_data["epoch"],
            y=fold_data["train_loss"],
            mode='lines',
            name='Train Loss',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=fold_data["epoch"],
            y=fold_data["val_loss"],
            mode='lines',
            name='Val Loss',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'Fold {selected_fold + 1} 训练进度',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )
        status = f"Fold {selected_fold+1} 当前 Epoch: {fold_data['epoch'][-1]+1}"
        return fig, status
    return go.Figure(), "等待训练数据更新..."

@app.callback(
    Output('final-results-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_final_results(n_intervals):
    if global_final_results:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'Fold {i+1}' for i in range(len(global_final_results["folds"]))],
            y=global_final_results["folds"],
            marker_color='rgb(55, 83, 109)',
            text=[f"{acc:.2%}" for acc in global_final_results["folds"]],
            textposition='outside',
        ))
        annotation_text = (f"Mean Accuracy: {global_final_results['mean']:.2%} ± {global_final_results['std']:.2%}")
        fig.update_layout(
            title='10-Fold Cross Validation Results',
            xaxis_title='Fold',
            yaxis_title='Accuracy',
            yaxis_tickformat=".2%",
            annotations=[dict(
                x=0.5,
                y=-0.12,
                showarrow=False,
                text=annotation_text,
                xref="paper",
                yref="paper"
            )]
        )
        return fig
    return go.Figure()


@app.callback(
    Output('layer-probs-graph', 'figure'),
    Input('fold-selector', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_layer_probs(selected_fold, n_intervals):
    if selected_fold in global_training_progress and global_training_progress[selected_fold]["layer_probs"]:
        probs_data = global_training_progress[selected_fold]["layer_probs"]
        num_layers = len(probs_data[0]['in_probs'])

        fig = make_subplots(
            rows=num_layers,
            cols=1,
            subplot_titles=[f"Layer {i + 1} Probability Dynamics" for i in range(num_layers)]
        )

        for layer in range(num_layers):
            # 合并输入输出概率曲线
            fig.add_trace(go.Scatter(
                x=list(range(len(probs_data))),
                y=[epoch['in_probs'][layer] for epoch in probs_data],
                name=f"Layer {layer + 1} In",
                line=dict(color='#636EFA'),
                mode='lines+markers',
                marker=dict(size=4)
            ), row=layer + 1, col=1)

            fig.add_trace(go.Scatter(
                x=list(range(len(probs_data))),
                y=[epoch['out_probs'][layer] for epoch in probs_data],
                name=f"Layer {layer + 1} Out",
                line=dict(color='#EF553B'),
                mode='lines+markers',
                marker=dict(size=4)
            ), row=layer + 1, col=1)

        # 优化可视化布局
        fig.update_layout(
            height=200 * num_layers,
            title_text="<b>Layer-wise Probability Dynamics</b>",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # 添加Y轴统一范围
        for i in range(num_layers):
            fig.update_yaxes(range=[0, 1], row=i + 1, col=1)

        return fig
    return go.Figure()


@app.callback(
    Output('similarity-graph', 'figure'),
    [Input('fold-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_similarity_graph(selected_fold, n):
    fig = go.Figure()

    # 异常数据检测
    current_data = global_training_progress.get(selected_fold, {}).get("similarity", [])
    if not current_data:
        fig.add_annotation(text="等待数据初始化...", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # 动态获取层数
    num_layers = len(current_data[0]) if current_data else 0
    valid_epochs = [epoch for epoch, data in enumerate(current_data) if any(not np.isnan(v) for v in data)]

    # 为每层创建轨迹
    colors = ['#e41a1c', '#377eb8', '#4daf4a'][:num_layers]  # 红蓝绿三色
    for layer in range(num_layers):
        y_values = [current_data[epoch][layer] if epoch < len(current_data) else np.nan
                    for epoch in valid_epochs]

        fig.add_trace(go.Scatter(
            x=valid_epochs,
            y=y_values,
            mode='lines+markers',
            name=f'Layer {layer + 1}',
            line=dict(color=colors[layer], width=2),
            marker=dict(size=6, opacity=0.8),
            hovertemplate='Epoch: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))

    # 自适应坐标轴
    fig.update_layout(
        title="节点特征差异动态（您的原始计算）",
        xaxis_title="训练轮次",
        yaxis_title="平均平方差异",
        template="plotly_white",
        hovermode="x unified",
        height=400,
        yaxis=dict(
            rangemode='tozero',  # 强制y轴包含零点
            tickformat=".2e"  # 科学计数法显示
        )
    )
    return fig

def run_experiment_in_thread():
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
        tau0=0.5,
        temp=0.01,
        env_num_layers=3,
        act_num_layers=1,
        env_model_type='MEAN_GNN',
        act_model_type='MEAN_GNN',
        gumbel_model_type='LIN',
        early_stop_patience=3000,
        early_stop_delta=0.001
    )
    experiment = Experiment(args)
    final_results = experiment.run()
    print(f"10折交叉验证结果:")
    print(f"平均准确率: {final_results['mean']:.2%} ± {final_results['std']:.2%}")
    print(f"最佳折叠: {final_results['max']:.2%}")
    print(f"最差折叠: {final_results['min']:.2%}")

if __name__ == "__main__":
    training_thread = threading.Thread(target=run_experiment_in_thread, daemon=True)
    training_thread.start()
    app.run_server(debug=True, use_reloader=False)