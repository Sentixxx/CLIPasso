
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 64)
        self.gcn2 = GraphConvolution(64, 32)
        self.linear = nn.Linear(32, output_dim*2)
    def forward(self, X, adj):
        X = self.gcn1(X, adj)
        X = F.relu(X)  # Graph convolution + ReLU
        X = self.gcn2(X, adj)  # Output layer
        X= self.linear(X)
        X = nn.Sigmoid()(X)
        return X