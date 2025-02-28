from layers.gcn import WeightedGCNConv
from layers.gin import WeightedGINConv
from layers.gnn import WeightedGNNConv
from layers.linear import GraphLinear

def get_component_list(in_dim, out_dim, hidden_dim, num_layers, model_type, mlp_func,device,bias = True):
    """
    获取组件列表
    :param device: 设备位置
    :param model_type: 组件类型
    :param in_dim: 输入层维度
    :param out_dim: 输出层维度
    :param hidden_dim: 隐藏层维度
    :param num_layers: 组件层数
    :param mlp_func: GIN所需要的多层感知机的函数
    :return: 组件列表
    """
    component_list = []
    dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    if model_type == 'GIN':
        for i in range(len(dim_list) - 1):
            component_list.append(WeightedGINConv(in_channels = dim_list[i], out_channels = dim_list[i + 1], mlp_func = mlp_func, bias=bias).to(device))
    elif model_type == 'GCN':
        for i in range(len(dim_list) - 1):
            component_list.append(WeightedGCNConv(in_channels = dim_list[i], out_channels = dim_list[i + 1], bias=bias).to(device))
    elif model_type == 'MEAN_GNN':
        for i in range(len(dim_list) - 1):
            component_list.append(WeightedGNNConv(in_channels = dim_list[i], out_channels = dim_list[i + 1], bias=bias, aggr='mean').to(device))
    elif model_type == 'SUM_GNN':
        for i in range(len(dim_list) - 1):
            component_list.append(WeightedGNNConv(in_channels = dim_list[i], out_channels = dim_list[i + 1], bias=bias, aggr='sum').to(device))
    elif model_type == 'LIN':
        for i in range(len(dim_list) - 1):
            component_list.append(GraphLinear(in_features=dim_list[i], out_features=dim_list[i + 1], bias=bias).to(device))
    return component_list
