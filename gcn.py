import torch
import torch.nn as nn

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)
        if self.use_bias:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x