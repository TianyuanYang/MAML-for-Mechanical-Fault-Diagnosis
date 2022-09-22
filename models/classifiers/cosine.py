import math
import torch
import torch.nn as nn
import utils

from .classifiers import register
from ..modules import *


@register('cosine')
class CosineClassifier(Module):
    def __init__(self, in_dim, n_way, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_way, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x, params=None):
        y = utils.compute_logits(x, self.proto, self.metric, self.temp)
        # print(y.shape)
        return y
