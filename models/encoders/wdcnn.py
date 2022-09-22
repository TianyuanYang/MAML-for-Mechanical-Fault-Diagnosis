from collections import OrderedDict

import torch.nn as nn

from .encoders import register
from ..modules import *

__all__ = ['wdcnn']


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding, bn_args):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding

        self.conv = Conv1d(in_channels, out_channels, kernal_size, stride, padding)
        self.bn = BatchNorm1d(out_channels, **bn_args)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, params=None, episode=None):
        out = self.conv(x, get_child_dict(params, 'conv'))
        out = self.bn(out, get_child_dict(params, 'bn'), episode)
        out = self.pool(self.relu(out))
        return out


class WDCNN(Module):
    def __init__(self, bn_args):
        super(WDCNN, self).__init__()

        episodic = bn_args.get('episodic') or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        bn_args_ep['episodic'] = True
        bn_args_no_ep['episodic'] = False
        bn_args_dict = dict()
        for i in [1, 2, 3, 4, 5]:
            if 'conv%d' % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep

        self.encoder = Sequential(OrderedDict([
            ('conv1', ConvBlock(1, 16, 64, 16, 1, bn_args_dict[1])),
            ('conv2', ConvBlock(16, 32, 3, 1, 1, bn_args_dict[2])),
            ('conv3', ConvBlock(32, 64, 2, 1, 1, bn_args_dict[3])),
            ('conv4', ConvBlock(64, 64, 3, 1, 1, bn_args_dict[4])),
            ('conv5', ConvBlock(64, 64, 3, 1, 0, bn_args_dict[5])),
        ]))

    def get_out_dim(self, scale=1):
        return 64 * scale

    def forward(self, x, params=None, episode=None):
        out = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        out = out.view(out.shape[0], -1)
        return out


@register('wdcnn')
def wdcnn(bn_args=dict()):
    return WDCNN(bn_args)
