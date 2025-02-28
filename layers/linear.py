from torch import Tensor
from torch.nn import Linear
from torch_geometric.typing import Adj, OptTensor

class GraphLinear(Linear):
    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        return super().forward(x)