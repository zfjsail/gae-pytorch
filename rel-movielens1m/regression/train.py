# Naive GAE for classification task in rel-movielens1M
# Paper: T. N. Kipf, M. Welling, Variational Graph Auto-Encoders ArXiv:1611.07308
# Test MSE Loss: 12.9219
# Runtime: 37.66
# Cost: N/A
# Description: Simply apply GAE to movielens. Graph was obtained by sampling from foreign keys. Features were llm embeddings from table data to vectors.

from __future__ import division
from __future__ import print_function
import torch
import argparse
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch import optim
from model import GAE_REGRESSION
import sys
sys.path.append("../../../../rllm/dataloader")

import time
from load_data import load_data
import networkx as nx

time_start = time.time()
# Define command-line arguments using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=50, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=8, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=4, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')

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


# Function to convert adjacency matrix to networkx graph
def change_to_matrix(adj):
    adj_sparse = adj.to_sparse()
    graph = nx.Graph()
    graph.add_nodes_from(range(adj_sparse.shape[0]))
    edges = adj_sparse.coalesce().indices().t().tolist()
    graph.add_edges_from(edges)
    adj = nx.adjacency_matrix(graph)
    return adj

def preprocess_graph(adj):
    """Preprocess the adjacency matrix for graph-based tasks."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def gae_for(args):
    print("Using {} dataset".format("movielens-regression"))
    # load movielens-regression dataset
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-regression')
    # print("adj_shape", adj.shape)
    # print("labels_shape", labels.shape)
    n_nodes, feat_dim = features.shape

    # convert adj to networkx graph
    adj = change_to_matrix(adj)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()


    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    # print("adj_norm_shape", adj_norm.shape)
    num_classes = labels.shape[0]

    # build the GAE_REGRESSION model and optimizer
    # print("build model")
    # model = GAE_REGRESSION(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model = GAE_REGRESSION(feat_dim, args.hidden1, args.hidden2, num_classes, args.dropout)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()

    # training loop
    for epoch in range(args.epochs):
        # print(epoch)
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        # recovered = model(features, adj_norm)
        loss = loss_function(recovered[idx_train], labels[idx_train])
        loss.backward()
        # cur_loss = loss.item()
        optimizer.step()

        # evaluate on validation set
        loss_val = loss_function(recovered[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    print("Optimization Finished!")
    time_end = time.time()
    print("Total time elapsed:", time_end - time_start)
    # test the model
    model.eval()
    recovered, mu, logvar = model(features, adj_norm)
    # recovered = model(features, adj_norm)
    loss_test = loss_function(recovered[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item())
          )


if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gae_for(args)
