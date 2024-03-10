from __future__ import division
from __future__ import print_function

import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelVAE, GAE_REGRESSION, GAE_CLASSIFICATION
# from optimizer import loss_function
from sklearn.metrics import f1_score
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import sys
sys.path.append("../../../../rllm/dataloader")

import time
from load_data import load_data
import networkx as nx

time_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=8, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=8, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

args = parser.parse_args()
def adj_matrix_to_list(adj_matrix):
    """
    This function converts adjacency matrices to adjacency lists
    Args:
        adj_matrix (COO Sparse Tensor): The adjacency matrix representing the connections between nodes.
    """
    adj_list = {}
    adj_matrix = adj_matrix.to_dense()
    for i in range(adj_matrix.size(0)):
        adj_list[i] = (
            adj_matrix[i] > 0
            ).nonzero(as_tuple=False).squeeze().tolist()
        # Ensure each value is a list, even if there's only one neighbor
        if not isinstance(adj_list[i], list):
            adj_list[i] = [adj_list[i]]
    return adj_list
def change_to_matrix(adj):
    # 获取邻接矩阵的稀疏表示
    adj_sparse = adj.to_sparse()
    # 将 PyTorch Tensor 转为 NetworkX 图对象
    graph = nx.Graph()
    # 添加节点
    graph.add_nodes_from(range(adj_sparse.shape[0]))
    # 添加边
    edges = adj_sparse.coalesce().indices().t().tolist()
    graph.add_edges_from(edges)
    # 获取邻接矩阵的密集表示
    adj = nx.adjacency_matrix(graph)
    return adj

def gae_for(args):
    print("Using {} dataset".format("movielens-regression"))
    # adj, features = load_data(args.dataset_str)
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-regression')
    # print("adj_shape", adj.shape)
    # print("labels_shape", labels.shape)
    n_nodes, feat_dim = features.shape
    # 获取邻接矩阵的稀疏表示
    adj = change_to_matrix(adj)

    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()


    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    # print("adj_norm_shape", adj_norm.shape)
    num_classes = labels.shape[0]
    model = GAE_REGRESSION(feat_dim, args.hidden1, args.hidden2, num_classes, args.dropout)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(recovered[idx_train], labels[idx_train])
        loss.backward()
        # cur_loss = loss.item()
        optimizer.step()

        # hidden_emb = mu.data.numpy()
        # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        loss_val = loss_function(recovered[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    print("Optimization Finished!")

    # test
    model.eval()
    recovered, mu, logvar = model(features, adj_norm)
    # hidden_emb = mu.data.numpy()
    loss_test = loss_function(recovered[idx_test], labels[idx_test])

    time_end = time.time()
    print("Total time elapsed:", time_end-time_start)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item())
          )


if __name__ == '__main__':
    gae_for(args)
