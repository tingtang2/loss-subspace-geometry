from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0.15):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x
