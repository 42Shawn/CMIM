import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class cmim(nn.Module):
    def __init__(self, fp_model, bnn_model, contrastive_alignment):
        super(cmim, self).__init__()

        t_channels = fp_model.get_channel_num()
        s_channels = bnn_model.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])
        self.avgpool = nn.AvgPool2d(8)

        fp_bns = fp_model.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in fp_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.fp_model = fp_model
        self.bnn_model = bnn_model
        self.contrastive_alignment = contrastive_alignment.cuda()

    def forward(self, x, idx, sample_idx):

        fp_feats = self.fp_model.extract_feature(x)
        binary_feats = self.bnn_model.extract_feature(x)

        loss_cmim = self.contrastive_alignment(fp_feats[-1], binary_feats[-1], idx, sample_idx)

        return loss_cmim
