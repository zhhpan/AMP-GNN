import torch

from lrgb.encoders.laplace import LapPENodeEncoder
from lrgb.encoders.mol_encoder import AtomEncoder


class Concat2NodeEncoder(torch.nn.Module):
    """Encoder that concatenates two node encoders.
    """

    def __init__(self, in_dim, emb_dim, enc2_dim_pe):
        super().__init__()
        # PE dims can only be gathered once the cfg is loaded.
        self.encoder1 = AtomEncoder(emb_dim)
        self.encoder2 = LapPENodeEncoder(dim_in=in_dim, dim_emb=emb_dim, expand_x=False)

    def forward(self, x, pestat):
        x = self.encoder1(x, pestat)
        x = self.encoder2(x, pestat)
        return x
