import torch
from torch import nn
import random,sys,time
from itertools import permutations, combinations
from torch_geometric.nn import GraphConv, SAGEConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, ELU, PReLU, LeakyReLU, Sigmoid, LayerNorm
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, double_layer=False):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_size, hidden_size, normalize=True, bias=False)
        torch.nn.init.xavier_uniform_(self.conv1.lin_l.weight)
        torch.nn.init.xavier_uniform_(self.conv1.lin_r.weight)
        self.conv2 = SAGEConv(hidden_size, hidden_size, normalize=True, bias=False)
        torch.nn.init.xavier_uniform_(self.conv2.lin_l.weight)
        torch.nn.init.xavier_uniform_(self.conv2.lin_r.weight)
        self.fc1 = Sequential(Linear(hidden_size+input_size, hidden_size), nn.LeakyReLU())
        self.fc2 = Sequential(Linear(hidden_size+hidden_size, hidden_size), nn.LeakyReLU())
        self.fc3 = Sequential(Linear(hidden_size, hidden_size), nn.LeakyReLU())
        self.act =  nn.LeakyReLU()
        self.double_layer = double_layer
    def forward(self, x, edge_index, batch):
        x = self.preprocess_features(x)
        x_ = x.clone()
        x = self.act(self.conv1(x, edge_index))
        x = torch.cat((x,x_),dim=1)
        x = self.fc1(x)
        if self.double_layer:
            x_ = x.clone()
            x = self.act(self.conv2(x, edge_index))
            x = torch.cat((x,x_),dim=1)
            x = self.fc2(x)
        x = global_add_pool(x, batch)
        x = self.fc3(x)
        x = F.normalize(x, dim=1, p=2)
        return x
    def preprocess_features(self, features):
        features = features.cpu().detach().numpy()
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        features = torch.tensor(features).float()
        if next(self.parameters()).get_device() == -1: return torch.tensor(features).float()
        return torch.tensor(features).float().to(next(self.parameters()).get_device())

class kGNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, aggr="add"):
        super(kGNN, self).__init__()
        self.conv1 = GraphConv(input_size, hidden_size, aggr=aggr)
        self.conv2 = GraphConv(hidden_size, hidden_size, aggr=aggr)
        self.conv3 = GraphConv(hidden_size, hidden_size, aggr=aggr)
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        #x = F.normalize(x, dim=1, p=2)
        return x
