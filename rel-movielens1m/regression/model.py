import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution

'''
class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

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
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar
'''



class GAE_CLASSIFICATION(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_classes, dropout):
        super(GAE_CLASSIFICATION, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.fc_out = nn.Linear(hidden_dim2, num_classes)

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
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        out = self.fc_out(z)
        return out, mu, logvar
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
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_classes, dropout):
        super(GAE_REGRESSION, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.regression_decoder = nn.Linear(hidden_dim2, num_classes)
        self.regression_decoder = RegressionDecoder(hidden_dim2, num_classes, dropout)

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