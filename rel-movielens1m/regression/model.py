import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RegressionDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, input_dim, output_dim, dropout):
        super(RegressionDecoder, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        output = self.linear(z)
        return output


class GAE_REGRESSION(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2,
                 num_classes, dropout):
        super(GAE_REGRESSION, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1,
                                    dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2,
                                    dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2,
                                    dropout, act=lambda x: x)
        # self.regression_decoder = nn.Linear(hidden_dim2, num_classes)
        self.regression_decoder = RegressionDecoder(hidden_dim2,
                                                    num_classes, dropout)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        # print("mu, logvar")
        mu, logvar = self.encode(x, adj)
        # print("reparameterize")
        h = logvar.mean(dim=0)
        # print("return")
        return self.regression_decoder(h), mu, logvar
