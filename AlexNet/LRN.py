import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable

class LRN(nn.Module):
    def __init__(self, k = 2, n = 5, alpha = 1e-4, beta = 0.75):
        super(LRN, self).__init__()
        self.k = k
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        pass


