import torch
import torch.nn as nn
import torch.nn.functional as F

from gae.layers import GraphConvolution


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=nn.Sigmoid()):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.sigmoid = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.sigmoid(torch.mm(z, z.t()))
        return adj