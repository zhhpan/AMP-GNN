## CoGNN

#### 一. 代码结构

<img src="https://gitee.com/zhhpan/pic-go_img/raw/master/images/image-20250228205102283.png" style="zoom:50%;" />

#### 二. 代码介绍

##### 1.  datasets/

​	用于存放数据集的文件夹

##### 2.  folds/

​	用于保存模型训练过程中，单次fold所产生的最佳模型，用于后面进行模型测试，命名以``{dataset.name}_fold{fold}.pth``为规则，其中``dataset.name``为测试数据集的名称，``fold``为对应fold的标号。

##### 3. layers/

​	用于存放最基本的gnn（gnn.pt），gcn（gcn.pt），gin（gin.pt），linear（linear.pt）模型，以及包含了一个加载这些模型的函数（存放于load_helper.pt 中）和用于gumbel_softmax的可学习的temp参数（temp.pt）

​	在gnn（gnn.pt），gcn（gcn.pt），gin（gin.pt）中，对``MessagePassing``进行了继承，并且重写了``forward``和``message``，添加了``edge_weight``参数，其在进行消息传递时对边产生影响。

##### 4.  network/

​	用于存放动作网络和环境网络

##### 5.  dataset.pt

​	自动加载 `HeterophilousGraphDataset` 中的指定数据集。提供 **10折交叉验证** 接口，通过 `select_fold_and_split` 方法按折叠索引划分训练/验证/测试集。动态获取特征维度（`num_features`）、输出类别数（`get_out_dim`）。

##### 6.  model.pt

​	定义了项目核心代码cognn，包含了初始化以及``forward``函数

##### 7.  param.pt

​	定义了环境网络、动作网络、Gumbel_Softmax所需要的参数，并进行了包装

##### 8.  main.pt

​	main函数中定义了模型的参数设置

​	experience类中定义了模型初始化，模型训练、验证、测试的整个流程，并给出可视化界面给出具体的数据

#### 三. 运行流程

**main.pt**

main函数  ->  参数处理  -> 加载数据集  ->  10折交叉验证  ->输出结果

**model.pt**

初始化参数 -> 环境网络中的编码器 -> 对环境网络中的组件的每一层（层归一化 -> 计算节点是否接收和传递消息的概率 -> 应用Gumbel_Softmax进行分类 ->  计算对应边的权重edge_weight -> 应用对应层的组件 -> 使用dropout和ReLU -> 输出x）->  应用层归一化  ->  环境网络中的解码器

#### 四. 运行需求

```bash
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-geometric==2.3.0
pip install torchmetrics ogb rdkit
pip install matplotlib
```
#### 五. 参数设置

```python
from argparse import Namespace
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
        env_model_type = 'MEAN_GNN',
        act_model_type = 'MEAN_GNN',
        gumbel_model_type = 'LIN'
    )
```

其中，``dataset_name``可以取值为``'roman-empire'``, ``'amazon-ratings'``, ``'minesweeper'``, ``'tolokers'``, ``'questions'``

​		    ``env_model_type``, ``act_model_type``可以取值为``'GIN'``,  ``'GCN'``,  ``'MEAN_GNN'``  ,``'SUM_GNN'``