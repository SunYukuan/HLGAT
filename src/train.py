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
from model import HLGAT 
import json
import nni

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='squirrel', 
                    help='cornell texas wisconsin chameleon squirrel')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--p_l', type=float, default=0.1, help='.')
parser.add_argument('--p_h', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.8, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()
if args.dataset == 'cora':
    jsontxt = open("../Param/param_cora.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'citeseer':
    jsontxt = open("../Param/param_citeseer.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'pubmed':
    jsontxt = open("../Param/param_pubmed.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'texas':
    jsontxt = open("../Param/param_texas.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'cornell':
    jsontxt = open("../Param/param_cornell.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'wisconsin':
    jsontxt = open("../Param/param_wisconsin.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'film':
    jsontxt = open("../Param/param_film.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'chameleon':
    jsontxt = open("../Param/param_chameleon.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'squirrel':
    jsontxt = open("../Param/param_squirrel.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'zinc':
    jsontxt = open("../Param/param_zinc.json", 'r').read()
    param = json.loads(jsontxt)
elif args.dataset == 'syn-relation':
    jsontxt = open("../Param/param_syn-relation.json", 'r').read()
    param = json.loads(jsontxt)
else:
    param = args.__dict__

param.update(nni.get_next_parameter())

if param['dataset_num'] == 0:
    param['dataset'] = 'texas'
    param['in_dim'] = 1703
if param['dataset_num'] == 1:
    param['dataset'] = 'cornell'
    param['in_dim'] = 1703
if param['dataset_num'] == 2:
    param['dataset'] = 'wisconsin'
    param['in_dim'] = 1703
if param['dataset_num'] == 3:
    param['dataset'] = 'film'
    param['in_dim'] = 932
if param['dataset_num'] == 4:
    param['dataset'] = 'chameleon'
    param['in_dim'] = 2325
if param['dataset_num'] == 5:
    param['dataset'] = 'squirrel'
    param['in_dim'] = 2089

param['save_mode'] = 0
# param['seed'] = args.seed
print("eps {:} | layer_num {:} | tr_ratio {:} | dropout {:}".format(
         param['eps'], param['layer_num'], param['train_ratio'], param['dropout']))
device = 'cuda'
accs = []
for i in range(20):
    g, nclass, features, labels, train, val, test = preprocess_data(param['dataset'], param['train_ratio'])
    features = features.to(device)
    labels = labels.to(device)
    rain = train.to(device)
    test = test.to(device)
    val = val.to(device)



    g = g.to(device)
    deg = g.in_degrees().cuda().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm

    net = HLGAT(g, features.size()[1], param['hidden'], nclass, param['dropout'], param['eps'], param['layer_num'], param['p_l'], param['p_h']).cuda()

    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
    print("eps {:} | layer_num {:} | tr_ratio {:} | dropout {:} | pl {:} | ph {:}|".format(
                param['eps'], param['layer_num'], param['train_ratio'], param['dropout'], param['p_l'], param['p_h']))
    # main loop
    dur = []
    los = []
    loc = []
    counter = 0
    min_loss = 100.0
    max_acc = 0.0

    for epoch in range(param['epochs']):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        logp = net(features)

        cla_loss = F.nll_loss(logp[train], labels[train])
        loss = cla_loss
        train_acc = accuracy(logp[train], labels[train])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        net.eval()
        logp = net(features)
        test_acc = accuracy(logp[test], labels[test])
        loss_val = F.nll_loss(logp[val], labels[val]).item()
        val_acc = accuracy(logp[val], labels[val])
        los.append([epoch, loss_val, val_acc, test_acc])

        if loss_val < min_loss and max_acc < val_acc:
            min_loss = loss_val
            max_acc = val_acc
            counter = 0
        else:
            counter += 1

        if counter >= param['patience'] and param['dataset'] in ['cora', 'citeseer', 'pubmed']:
            print('early stop')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        #print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
            #epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

    torch.save(net.state_dict(), param['dataset']+'-net-{}.pth'.format(i))
    #torch.save(net, 'net1.pt')
    print('saveing finished')
    if param['dataset'] in ['cora', 'citeseer', 'pubmed'] or 'syn' in param['dataset']:
        los.sort(key=lambda x: x[1])
        acc = los[0][-1]
        print(acc)
        accs.append(acc)
    else:
        los.sort(key=lambda x: -x[2])
        acc = los[0][-1]
        print(acc)
        accs.append(acc)
    #torch.save(net,'====\n{} - {} - {}.pth'.format(param['dataset'], param['train_ratio'], param['layer_num']))
print('==========================', np.mean(accs))