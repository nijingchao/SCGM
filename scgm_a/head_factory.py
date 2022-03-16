import torch
import numpy as np
from torch import nn

HEADS = ['seq', 'fork', 'cls', 'none', 'e', 'f', 'fork1', 'seq1', 'seq2', 'fork2', 'fork3']


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    # init = init / (init.norm(2, dim=1).unsqueeze(1) + 1e-8)
    return init


class HeadFactory(object):
    def create_head(self, feature_dim, dim, head_type, num_classes, num_subclasses, norm, with_mlp=True):
        hidden_size = 2048
        if head_type == 'seq':
            # 2048 => 128 => #C
            head_q = SequentialFC(feature_dim, dim, num_classes)
            head_k = nn.Linear(feature_dim, dim)
            head_mapping = {'weight': 'fc1.weight', 'bias': 'fc1.bias'}
        elif head_type == 'seq_em':
            # 2048 => 128
            head_q = SequentialEMFC(feature_dim, dim, num_classes, num_subclasses, norm)
            head_k = nn.Linear(feature_dim, dim)
            head_mapping = {'weight': 'fc1.weight', 'bias': 'fc1.bias'}
        elif head_type == 'fork':
            # 2048 => 128
            #      \=> #C
            head_q = ForkFC(feature_dim, dim, num_classes)
            head_k = nn.Linear(feature_dim, dim)
            head_mapping = {'weight': 'fc1.weight', 'bias': 'fc1.bias'}
        elif head_type == 'cls':
            # 2048 => 128
            head_q = nn.Linear(feature_dim, num_classes)
            head_k = nn.Linear(feature_dim, num_classes)
            head_mapping = None
        elif head_type == 'none':
            head_q = nn.Linear(feature_dim, dim)
            head_k = nn.Linear(feature_dim, dim)
            head_mapping = {'weight': 'weight', 'bias': 'bias'}
        else:
            raise NotImplementedError
        if with_mlp:
            # 2048 => 2048 => ...
            head_q = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), head_q)
            head_k = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), head_k)
            if head_mapping is not None:
                mapping = {'0.weight': '0.weight', '0.bias': '0.bias'}
                head_mapping.update(mapping)
                if 'weight' in head_mapping and 'bias' in head_mapping:
                    head_mapping['2.weight'] = '2.' + head_mapping.pop('weight')
                    head_mapping['2.bias'] = '2.' + head_mapping.pop('bias')
        return head_q, head_k, head_mapping


class ForkFC(nn.Module):
    def __init__(self, in_dim, out_dim1, out_dim2):
        super(ForkFC, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim1)
        self.fc2 = nn.Linear(in_dim, out_dim2)

    def forward(self, inputs):
        out1 = self.fc1(inputs)
        out2 = self.fc2(inputs)
        return out1, out2


class SequentialFC(nn.Module):
    def __init__(self, in_dim, dim1, dim2):
        super(SequentialFC, self).__init__()
        self.fc1 = nn.Linear(in_dim, dim1)
        self.fc2 = nn.Linear(dim1, dim2)

    def forward(self, inputs):
        out1 = self.fc1(inputs)
        out2 = self.fc2(nn.functional.relu(out1))
        return out1, out2


class SequentialEMFC(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, num_subclasses, norm=False):
        super(SequentialEMFC, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.mu_y = nn.Parameter(glorot([num_classes, hid_dim]), requires_grad=True)
        self.mu_z = nn.Parameter(glorot([num_subclasses, hid_dim]), requires_grad=True)
        self.norm = norm

    def forward(self, x):
        out1 = self.fc1(x)
        if self.norm is True:
            out2 = nn.functional.normalize(out1, p=2, dim=1)
            out2 = (nn.functional.relu(out2)) @ (self.mu_y.t())
        else:
            out2 = (nn.functional.relu(out1)) @ (self.mu_y.t())
        return out1, out2
