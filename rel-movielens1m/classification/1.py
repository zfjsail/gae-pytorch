from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
import sys
sys.path.append("../../../../rllm/dataloader")
from load_data import load_data
from model import GAE_CLASSIFICATION

t_total = time.time()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units in layer 1.')
parser.add_argument('--hidden2', type=int, default=8,
                    help='Number of hidden units in layer 2.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification')

# Model and optimizer
num_classes = labels.shape[1]
model = GAE_CLASSIFICATION(input_feat_dim=features.shape[1],
                            hidden_dim1=args.hidden1,
                            hidden_dim2=args.hidden2,
                            num_classes=num_classes,
                            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

loss_function = nn.BCEWithLogitsLoss()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    recovered, mu, logvar = model(features, adj)
    loss = loss_function(recovered[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    hidden_emb = mu.data.numpy()

    # 计算训练集的F1分数
    pred_train = np.where(recovered[idx_train].detach().numpy() > -1.0, 1, 0)
    f1_micro_train = f1_score(labels[idx_train].detach().numpy(), pred_train, average="micro")
    f1_macro_train = f1_score(labels[idx_train].detach().numpy(), pred_train, average="macro")

    # roc_curr, ap_curr = get_roc_score(hidden_emb, adj, idx_val, idx_test)

    print('Epoch: {:04d}'.format(epoch + 1),
          'train_loss: {:.4f}'.format(loss.item()),
          'f1_train_micro: {:.4f}'.format(f1_micro_train),
          'f1_train_macro: {:.4f}'.format(f1_macro_train),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    recovered, mu, logvar = model(features, adj)
    hidden_emb = mu.data.numpy()

    # 计算测试集的F1分数
    pred_test = np.where(recovered[idx_test].detach().numpy() > -1.0, 1, 0)
    f1_micro_test = f1_score(labels[idx_test].detach().numpy(), pred_test, average="micro")
    f1_macro_test = f1_score(labels[idx_test].detach().numpy(), pred_test, average="macro")

    # roc_score, ap_score = get_roc_score(hidden_emb, adj, idx_test)


    print("Test F1 micro: {:.4f}".format(f1_micro_test))
    print("Test F1 macro: {:.4f}".format(f1_macro_test))

# Train model
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")

# Testing
test()