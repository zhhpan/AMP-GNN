import logging
import os
import os.path as osp
from argparse import Namespace
from typing import Any, Dict, Tuple
import time
import numpy as np
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torchmetrics import Accuracy, AUROC, AveragePrecision
from tqdm import tqdm
from dataset import DataSet as DS
from lrgb.cosine_scheduler import cosine_with_warmup_scheduler
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
        if(self.dataset_name == 'func'):
            self.task_loss = BCEWithLogitsLoss()
        else:
            self.task_loss = CrossEntropyLoss()

        # 折叠信息
        self.folds = self.dataset.get_folds()
        self.num_classes = self.dataset.get_out_dim()
        self.logger.info(f"分类类别数: {self.num_classes}")

        # 评估指标
        if self.dataset_name in ['roman_empire', 'amazon_ratings']:
            self.accuracy = Accuracy(
                task="multiclass",
                num_classes=self.num_classes
            ).to(self.device)
        elif self.dataset_name in ['minesweeper', 'tolokers', 'questions']:
            self.accuracy = AUROC(
                task="multiclass",
                num_classes=self.num_classes
            ).to(self.device)
        elif self.dataset_name in ['func']:
            self.accuracy = AveragePrecision(
                task="multilabel",
                num_labels=self.num_classes
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
            is_lrgb=True if self.dataset_name == 'func' else False
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
        if(self.dataset_name == 'func'):
            self.dataset.data.y = self.dataset.data.y.to(dtype=torch.float)

    def create_data_loaders(self) -> Dict[str, DataLoader]:
        def apply_mask(data: Data, mask_type: str) -> Data:
            masked_data = data
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
        edge_attr = data.edge_attr
        if data.edge_attr is not None:
            edge_attr = edge_attr.to(device=self.device)
        out = model(x = data.x.to(self.device), edge_index = data.edge_index.to(self.device), edge_attr = edge_attr, pestat = [data.EigVals.to(self.device), data.EigVecs.to(self.device)])
        return self.task_loss(out[node_mask], data.y.to(self.device)[node_mask])

    def evaluate(self, model: torch.nn.Module, loader: DataLoader, mask: str) -> Tuple[float, float]:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                loss = self.calculate_loss(model, data, mask)
                total_loss += loss.item()
                edge_attr = data.edge_attr
                if data.edge_attr is not None:
                    edge_attr = edge_attr.to(device=self.device)
                pred = model(x = data.x.to(self.device), edge_index = data.edge_index.to(self.device), edge_attr = edge_attr, pestat = [data.EigVals.to(self.device), data.EigVecs.to(self.device)]).argmax(dim=1)
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
            if (self.dataset_name == 'func'):
                env_edge_embedding = BondEncoder(env_params.env_dim)
                act_edge_embedding = BondEncoder(action_params.hidden_dim)
            model = AMPGNN(gumbel_params, env_params, action_params, self.device, self.use_model, env_edge_embedding,act_edge_embedding).to(self.device)
            if (self.dataset_name == 'func'):
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=5e-4)
                scheduler = cosine_with_warmup_scheduler(optimizer=optimizer,num_warmup_epochs=1000,max_epoch=self.epochs)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
                scheduler = None


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
                if scheduler:
                    scheduler.step(val_loss)
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
        final_results = {
            "folds": test_accs.tolist(),
            "mean": test_accs.mean().item(),
            "std": test_accs.std().item(),
            "max": test_accs.max().item(),
            "min": test_accs.min().item()
        }
        # global_final_results = final_results
        # print('global_final_results:', global_final_results)
        return final_results


def run_experiment_in_thread():
    global  global_final_results
    args = Namespace(
        dataset_name='func',
        seed=0,
        batch_size=2,
        env_dim=128,
        act_dim=16,
        dropout=0.2,
        lr=0.001,
        epochs=3000,
        learn_temp=True,
        tau0=0.5,
        temp=0.01,
        env_num_layers=2,
        act_num_layers=1,
        env_model_type='MEAN_GNN',
        act_model_type='MEAN_GNN',
        gumbel_model_type='LIN',
        early_stop_patience=200,
        early_stop_delta=0.001,
        use_model = True,
    )
    experiment = Experiment(args)
    final_results = experiment.run()
    global_final_results = final_results
    print(f"10折交叉验证结果:")
    print(f"平均准确率: {final_results['mean']:.2%} ± {final_results['std']:.2%}")
    print(f"最佳折叠: {final_results['max']:.2%}")
    print(f"最差折叠: {final_results['min']:.2%}")

if __name__ == "__main__":
    # 单独运行时启动训练
    run_experiment_in_thread()