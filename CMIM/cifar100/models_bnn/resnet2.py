'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *


__all__ =['resnet18A_1w1a','resnet18B_1w1a','resnet18C_1w1a','resnet18_1w1a','resnet18_fp','resnet34_fp','resnet50_fp']
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class Bottleneck_1w1a(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out

    def get_channel_num(self):

        return [16, 32, 64, 64]

    def extract_feature(self, x, preReLU=True):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        out = self.layer4(feat3)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat4 = self.bn2(out)

        if not preReLU:
            # feat1 = F.relu(feat1)
            # feat2 = F.relu(feat2)
            # feat3 = F.relu(feat3)
            feat1 = F.hardtanh(feat1)
            feat2 = F.hardtanh(feat2)
            feat3 = F.hardtanh(feat3)

        return [feat1, feat2, feat3, feat4] #, out

    def get_bn_before_relu(self):
        # if isinstance(self.layer1[0], Bottleneck):
        #     bn1 = self.layer1[-1].bn3
        #     bn2 = self.layer2[-1].bn3
        #     bn3 = self.layer3[-1].bn3
        # elif isinstance(self.layer1[0], BasicBlock):
        #     bn1 = self.layer1[-1].bn2
        #     bn2 = self.layer2[-1].bn2
        #     bn3 = self.layer3[-1].bn2
        # else:
        #     print('ResNet unknown block error !!!')
        bn1 = self.layer1[-1].bn2
        bn2 = self.layer2[-1].bn2
        bn3 = self.layer3[-1].bn2

        return [bn1, bn2, bn3]



class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        super(ResNet2, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out, f4_pre = self.layer4(out)
        f4 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out]
            else:
                return [f0, f1, f2, f3, f4, f5], out
        else:
            return out

    def get_channel_num(self):

        return [16, 32, 64]

    def extract_feature(self, x, preReLU=True):

        x = self.conv1(x)
        x = self.bn1(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)

        # x = F.relu(feat3)
        # # x = self.avgpool(x)
        # x = nn.AvgPool2d(8)(x)
        # x = x.view(x.size(0), -1)
        # out = self.linear(x)

        if not preReLU:
            # feat1 = F.relu(feat1)
            # feat2 = F.relu(feat2)
            # feat3 = F.relu(feat3)
            feat1 = F.hardtanh(feat1)
            feat2 = F.hardtanh(feat2)
            feat3 = F.hardtanh(feat3)

        return [feat1, feat2, feat3] #, out




def resnet18A_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18B_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[32,64,128,256],**kwargs)

def resnet18C_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,128,256,512],**kwargs)


def resnet18_fp(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], [64,128,256,512], **kwargs)

def resnet34_fp():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet34():
    return ResNet(BasicBlock_1w1a, [3,4,6,3])


def resnet50_fp():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck_1w1a, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck_1w1a, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck_1w1a, [3,8,36,3])


def test():
    net = resnet18_1w1a()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
