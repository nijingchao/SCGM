import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    # init = init / (init.norm(2, dim=1).unsqueeze(1) + 1e-8)
    return init


class mlp_classifier(nn.Module):
    def __init__(self, dim, num_class):
        super(mlp_classifier, self).__init__()
        self.fc = nn.Linear(dim, num_class, bias=False)

    def loss(self, logit, y, w=None):
        ls = F.cross_entropy(logit, y, weight=w)
        return ls

    def forward(self, x):
        output = self.fc(x)
        return output


class identity_layer(nn.Module):
    def __init__(self):
        super(identity_layer, self).__init__()

    def forward(self, x):
        return x


class mlp_classifier_2(nn.Module):
    def __init__(self, k, num_class, z):
        super(mlp_classifier_2, self).__init__()
        self.fc = nn.Linear(k, num_class, bias=True)
        self.z = z.t()  # (dim, k)

    def loss(self, logit, y, w=None):
        ls = F.cross_entropy(logit, y, weight=w)
        return ls

    def forward(self, x):
        output = x @ (self.fc(self.z))  # (n, num_class)
        return output


class mlp_classifier_3(nn.Module):
    def __init__(self, k, num_class, z):
        super(mlp_classifier_3, self).__init__()
        self.w = nn.Parameter(glorot([k, num_class]), requires_grad=True)
        self.b = nn.Parameter(glorot([1, num_class]), requires_grad=True)
        self.z = z.t()  # (dim, k)

    def loss(self, logit, y, w=None):
        ls = F.cross_entropy(logit, y, weight=w)
        return ls

    def forward(self, x):
        w = torch.softmax(self.w, dim=0)
        output = x @ (self.z @ w)
        return output
