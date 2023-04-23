from torch import nn
from torch.nn import functional as F
from models.subspace_layers import SubspaceNonLinear, LinesLinear


class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout_prob=0.15):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class NN(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, dropout_prob) -> None:
        super().__init__()

        self.mlp = MLP(n_in=input_dim,
                       n_out=hidden_dim,
                       dropout_prob=dropout_prob)
        self.out = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.mlp(x)

        return self.out(x)


class SubspaceMLP(nn.Module):

    def __init__(self, n_in, n_out, seed, dropout_prob=0.15):
        super().__init__()

        self.linear = LinesLinear(n_in, n_out)
        self.linear.initialize(seed)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class SubspaceNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, dropout_prob,
                 seed) -> None:
        super().__init__()

        self.mlp = SubspaceMLP(n_in=input_dim,
                               n_out=hidden_dim,
                               dropout_prob=dropout_prob,
                               seed=seed)
        self.out = LinesLinear(in_features=hidden_dim, out_features=out_dim)
        self.out.initialize(seed)

    def forward(self, x):
        x = self.mlp(x)

        return self.out(x)
