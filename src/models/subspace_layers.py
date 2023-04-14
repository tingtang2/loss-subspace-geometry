# implementation from https://github.com/vaseline555/SuPerFed/blob/main/src/models/layers.py

from torch import nn
from torch.nn import functional as F
import torch


# Linear layer implementation
class SubspaceLinear(nn.Linear):

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.linear(input=x, weight=w, bias=self.bias)
        return x


class TwoParamLinear(SubspaceLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, seed):
        if seed == -1:  # SCAFFOLD
            torch.nn.init.zeros_(self.weight_1)
        else:
            torch.manual_seed(seed)
            torch.nn.init.xavier_normal_(self.weight_1)


class LinesLinear(TwoParamLinear):

    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight_1
        return w
