import torch
import torch.nn as nn

class GCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            bound = 1 / (self.in_channels**0.5)
            self.bias = nn.parameter.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.tensor, edge_index: torch.tensor):
        if not (hasattr(self, "coef") and self.cached):
            self.coef = self.compute_coef(edge_index)
        ret = self.coef @ self.linear(x)
        if self.bias is not None:
            ret += self.bias
        return ret

    def compute_coef(self, edge_index: torch.tensor):
        hat_A = edge_index
        if self.normalize:
            if self.add_self_loops:
                hat_A += torch.eye(edge_index.shape[0])
                if self.improved:
                    hat_A += torch.eye(edge_index.shape[0])
            tilde_D = 1 / torch.sqrt(torch.sum(hat_A, 1))
            hat_A = tilde_D.reshape(-1, 1) * tilde_D * hat_A
        return hat_A
