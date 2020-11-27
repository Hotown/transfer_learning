import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.domain_cls = nn.Sequential()
        self.domain_cls.add_module('d_fc1', nn.Linear(input_dim, hidden_dim))
        self.domain_cls.add_module('d_bn1', nn.BatchNorm1d(hidden_dim))
        self.domain_cls.add_module('d_relu1', nn.ReLU(True))
        self.domain_cls.add_module('d_fc2', nn.Linear(hidden_dim, 2))
        self.domain_cls.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.domain_cls(x)
        return out
