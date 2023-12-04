import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from utils import accuracy, preprocess_data
from model1 import FAGCN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='chameleon', 
                    help='cornell texas wisconsin chameleon squirrel')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--p_l', type=float, default=0.19, help='.')
parser.add_argument('--p_h', type=float, default=0.5, help='Dropout rate (1 - keep probability).')


parser.add_argument('--eps', type=float, default=0.8, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()
device = 'cuda'
los = []
g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, args.train_ratio)
features = features.to(device)
labels = labels.to(device)
rain = train.to(device)
test = test.to(device)
val = val.to(device)

g = g.to(device)
deg = g.in_degrees().cuda().float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm

net = FAGCN(g, features.size()[1], args.hidden, nclass, args.dropout, args.eps, args.layer_num, args.p_l, args.p_h).cuda()
net.load_state_dict(torch.load('data-chameleon__net-283.pth'))
#net = torch.load('net1.pt')
net.eval()
logp = net(features)
test_acc = accuracy(logp[test], labels[test])
print(test_acc)
loss_val = F.nll_loss(logp[val], labels[val]).item()
val_acc = accuracy(logp[val], labels[val])
los.append([loss_val, val_acc, test_acc])






