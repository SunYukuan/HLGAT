import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from dgl import function as fn
import numpy as np
import torch
import numpy as np


class HLLayer(nn.Module):
    def __init__(self, g, in_dim, dropout, p_l, p_h):
        super(HLLayer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate_low = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate_low.weight, gain=1.414)
        self.gate_high = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate_high.weight, gain=1.414)
        self.WRL = nn.Linear(2 * in_dim, in_dim)
        nn.init.xavier_normal_(self.WRL.weight, gain=1.414)
        self.p_l=p_l
        self.p_h=p_h

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        
        _low = self.gate_low(h2)
        g_low = F.relu(torch.where(_low>0, _low, -self.p_l*_low)).squeeze()
        e_low = g_low * edges.dst['d'] * edges.src['d']
        e_low = self.dropout(e_low)
        
        _high = self.gate_high(h2)
        g_high = -F.relu(torch.where(_high>0, _high, -self.p_h*_high)).squeeze()
        e_high = g_high * edges.dst['d'] * edges.src['d']
        b = (edges.dst['d'].cpu()).numpy()
        e_high = self.dropout(e_high)
        return {'e_low': e_low, 'm_low': g_low, 'e_high': e_high, 'm_high': g_high}

    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e_low', '_low'), fn.sum('_low', 'z_low'))
        self.g.update_all(fn.u_mul_e('h', 'e_high', '_high'), fn.sum('_high', 'z_high'))
        return self.WRL(torch.cat([self.g.ndata['z_low'], self.g.ndata['z_high']], dim=1))

class HLGAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num, p_l, p_h):
        super(HLGAT, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(HLLayer(self.g, hidden_dim, dropout, p_l, p_h))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)

