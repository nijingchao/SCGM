import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    # init = init / (init.norm(2, dim=1).unsqueeze(1) + 1e-8)
    return init


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            num_subclasses: int = 1000,
            kd_t: float = 4.0,
            hiddim: int=128,
            with_mlp: bool = True,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # add parameters
        # ---
        feat_dim = 512 * block.expansion
        if with_mlp is True:
            self.fc_enc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, hiddim))
        else:
            self.fc_enc = nn.Linear(feat_dim, hiddim)

        self.mu_y = nn.Parameter(glorot([num_classes, hiddim]), requires_grad=True)
        self.mu_z = nn.Parameter(glorot([num_subclasses, hiddim]), requires_grad=True)
        self.hiddim = hiddim
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(kd_t)
        # ---

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def _embed_impl(self, x: Tensor) -> Tensor:
        x = self.fc_enc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def embed(self, x: Tensor) -> Tensor:
        return self._embed_impl(x)

    def loss(self, logit, q, y, tau, alpha, logit_t1=None, logit_t2=None, logit_t3=None, beta1=1.0, beta2=1.0, beta3=1.0, ang_norm=False, norm_type='logit'):
        '''
        :param logit: (n, hiddim)
        :param q: (n, k)
        :param y: (n, num_class)
        :param tau:
        :param alpha:
        :return:
        '''
        n = logit.shape[0]

        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)
        logit_norm = F.normalize(logit, p=2, dim=1)

        # # w/o angular norm
        # # ---
        # logit1 = logit_norm @ (mu_z.t())  # (n, k)
        # ls1 = torch.exp(logit1 / tau)  # (n, k)
        # ls1 = ls1 / ls1.sum(1).view(-1, 1)  # (n, k)
        # ls1 = - torch.log(ls1) * q  # (n, k)
        # ls1 = ls1.sum() / n

        if ang_norm is True:
            y_sample = y @ mu_y  # (n, d)
            logit1 = F.normalize(logit_norm - y_sample, p=2, dim=1)  # (n, d)
            logit2 = mu_z.t().unsqueeze(0) - y_sample.unsqueeze(-1)  # (n, d, k)
            logit2 = F.normalize(logit2, p=2, dim=1)  # (n, d, k)
            logit1 = (logit1.unsqueeze(-1) * logit2).sum(1)  # (n, k)
            logit1 = logit1 / tau  # (n, k)
        else:
            logit1 = logit_norm @ (mu_z.t())  # (n, k)
            logit1 = logit1 / tau  # (n, k)

        ls1 = self.criterion_cls(logit1, q.argmax(1))

        # loss on z
        # ---
        logit2 = (y @ mu_y) @ (mu_z.t())  # (n, k)
        ls2_num = torch.exp(logit2)  # (n, k)
        ls2_den = torch.exp(mu_y @ (mu_z.t()))  # (c, k)
        # ls2_num = torch.exp((y @ mu_y_nm) @ (mu_z.t()) + y @ self.mu_y_bias.view(-1, 1))  # (n, k)
        # ls2_den = torch.exp(mu_y_nm @ (mu_z.t()) + self.mu_y_bias.view(-1, 1))  # (c, k)

        # if w is not None:
        #     ls2_num = ls2_num * (y @ w.view(-1, 1))  # (n, k)
        #     ls2_den = ls2_den * w.view(-1, 1)  # (c, k)

        ls2 = - torch.log(ls2_num / ls2_den.sum(0).view(1, -1)) * q  # (n, k)
        ls2 = ls2.sum() / n

        if norm_type == 'logit':
            logit3 = (F.relu(logit_norm)) @ (self.mu_y.t())  # (n, num_class)
        elif norm_type == 'weight':
            logit3 = (F.relu(logit)) @ (mu_y.t())  # (n, num_class)
        elif norm_type == 'logit_and_weight':
            logit3 = (F.relu(logit_norm)) @ (mu_y.t())  # (n, num_class)
        elif norm_type == 'none':
            logit3 = (F.relu(logit)) @ (self.mu_y.t())  # (n, num_class)
        else:
            raise NotImplementedError

        ls3 = self.criterion_cls(logit3, y.argmax(1))
        # ls3 = F.cross_entropy(logit3, y.argmax(1), weight=w)

        if beta1 == 1.0:
            ls_div1 = 0.0
        else:
            ls_div1 = self.criterion_div(logit1, logit_t1)

        if beta2 == 1.0:
            ls_div2 = 0.0
        else:
            ls_div2 = self.criterion_div(logit2, logit_t2)

        if beta3 == 1.0:
            ls_div3 = 0.0
        else:
            ls_div3 = self.criterion_div(logit3, logit_t3)

        ls = alpha * (beta1 * ls1 + beta2 * ls2) + beta3 * ls3 + (1 - beta3) * ls_div3 + (1 - beta1) * ls_div1 + (1 - beta2) * ls_div2

        return ls, ls1, ls2, ls3, ls_div1, ls_div2, ls_div3

    def pred(self, x, tau):
        '''
        :param x: (n, hiddim)
        :param tau:
        :return:
        '''
        x = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        prob_z_x = torch.exp((x @ (mu_z.t())) / tau)  # (n, k)
        prob_z_x = prob_z_x / prob_z_x.sum(1).view(-1, 1)  # (n, k)

        prob_y_z = torch.exp((mu_z @ mu_y.t()))  # (k, c)
        prob_y_z = prob_y_z / prob_y_z.sum(1).view(-1, 1)  # (k, c)

        prob_y_x = prob_z_x @ prob_y_z

        return prob_y_x, prob_z_x, prob_y_z

    def forward_to_prob(self, x, y, tau):
        '''
        :param x: (n, hiddim)
        :param y: (n, num_class)
        :param tau:
        :return:
        '''
        x = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        prob_z_x = torch.exp((x @ (mu_z.t())) / tau)  # (n, k)
        prob_z_x = prob_z_x / prob_z_x.sum(1).view(-1, 1)  # (n, k)

        prob_y_z_num = torch.exp((y @ mu_y) @ (mu_z.t()))  # (n, k)
        prob_y_z_den = torch.exp(mu_y @ (mu_z.t()))  # (c, k)
        prob_y_z = prob_y_z_num / prob_y_z_den.sum(0).view(1, -1)

        prob_y_x = prob_z_x * prob_y_z  # (n, k)

        return prob_y_x, prob_y_z, prob_z_x

    def forward_to_logits(self, x, y, tau=0.1, norm_type='logit'):
        '''
        :param x: (n, hiddim)
        :param y: (n, num_class)
        :param tau:
        :param norm_type:
        :return:
        '''
        x_norm = F.normalize(x, p=2, dim=1)
        mu_z = F.normalize(self.mu_z, p=2, dim=1)
        mu_y = F.normalize(self.mu_y, p=2, dim=1)

        logit1 = x_norm @ (mu_z.t())  # (n, k)
        logit1 = logit1 / tau

        logit2 = (y @ mu_y) @ (mu_z.t())  # (n, k)

        if norm_type == 'logit':
            logit3 = (F.relu(x_norm)) @ (self.mu_y.t())  # (n, num_class)
        elif norm_type == 'weight':
            logit3 = (F.relu(x)) @ (mu_y.t())  # (n, num_class)
        elif norm_type == 'logit_and_weight':
            logit3 = (F.relu(x_norm)) @ (mu_y.t())  # (n, num_class)
        elif norm_type == 'none':
            logit3 = (F.relu(x)) @ (self.mu_y.t())  # (n, num_class)
        else:
            raise NotImplementedError

        return logit1, logit2, logit3


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
