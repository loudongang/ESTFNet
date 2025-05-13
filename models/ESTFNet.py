import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights


class ESTCA(nn.Module):
    """高效时空交叉注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 共享的压缩激发单元
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        # 轻量级空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # 使用深度卷积优化计算
            nn.Sigmoid()
        )

        # 双向交互卷积
        self.inter_conv = nn.Conv2d(in_channels * 2, 2, 3, padding=1)

    def forward(self, x1, x2):
        # 联合特征提取
        x_cat = torch.cat([x1, x2], dim=1)

        # 生成交互权重图
        gate = torch.sigmoid(self.inter_conv(x_cat))
        g1, g2 = gate.chunk(2, dim=1)

        # 通道协同校准
        sc = self.se(x1 + x2)

        # 空间协同注意力
        x_avg = torch.mean(x_cat, dim=1, keepdim=True)
        x_max, _ = torch.max(x_cat, dim=1, keepdim=True)
        sp = self.spatial(torch.cat([x_avg, x_max], dim=1))

        # 双向增强
        x1_out = x1 * (1 + sc * sp) + g1 * x2
        x2_out = x2 * (1 + sc * sp) + g2 * x1

        return x1_out, x2_out

class BDAF(nn.Module):
    """双向差分自适应融合"""

    def __init__(self, in_channels=128, ratio=8):
        super().__init__()
        # 差异感知卷积
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 3, padding=1)
        )

        # 双向注意力门
        self.att_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, 1),
            nn.Sigmoid()
        )

        # 特征校准单元
        self.calibrator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        # 动态差异建模
        diff_map = self.diff_conv(torch.abs(x1 - x2))

        # 双向门控融合
        att1 = self.att_gate(torch.cat([x1, diff_map], dim=1))
        att2 = self.att_gate(torch.cat([x2, diff_map], dim=1))

        # 特征校准
        x1_cal = self.calibrator(x1 * att1 + diff_map * (1 - att1))
        x2_cal = self.calibrator(x2 * att2 + diff_map * (1 - att2))

        # 梯度反传增强
        x1_final = x1 + (x2_cal.detach() - x1_cal).abs()
        x2_final = x2 + (x1_cal.detach() - x2_cal).abs()

        return x1_final, x2_final

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low // scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True))
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class base(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(base, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)
        self.Dec1 = _DecoderBlock(128, 64, 128)
        self.Dec2 = _DecoderBlock(128, 64, 128)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)
        self.DecCD = _DecoderBlock(128, 128, 128, scale_ratio=2)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        # self.stca = STCA(128)
        self.estca = ESTCA(128)
        self.bdaf_high = BDAF(128)  # 高层特征校准
        self.bdaf_low = BDAF(64)  # 低层特征校准

        initialize_weights(self.Dec1, self.Dec2, self.classifier1, self.classifier2, self.resCD, self.DecCD,
                           self.classifierCD, self.estca, self.bdaf_high, self.bdaf_low)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):

        x = self.FCN.layer0(x)  # size:1/2
        x = self.FCN.maxpool(x)  # size:1/4
        x_low = self.FCN.layer1(x)  # size:1/4
        # 低层特征校准
        x_low, _ = self.bdaf_low(x_low, x_low)  # 单时相自校准
        x = self.FCN.layer2(x_low)  # size:1/8
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x = self.FCN.head(x)
        # 高层特征校准
        x, _ = self.bdaf_high(x, x)

        return x, x_low

    def CD_forward(self, x1, x2):
        b, c, h, w = x1.size()
        # 跨时相高层校准
        x1, x2 = self.bdaf_high(x1, x2)
        # 时空-语义协同增强
        x1, x2 = self.estca(x1, x2)
        x = torch.cat([x1, x2], 1)
        xc = self.resCD(x)
        return xc

    def forward(self, x1, x2):
        x_size = x1.size()

        # 双时相特征提取
        x1, x1_low = self.base_forward(x1)
        x2, x2_low = self.base_forward(x2)


        # 变化特征提取
        xc = self.CD_forward(x1, x2)

        # 解码过程
        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)


        # 变化检测解码
        xc_low = torch.cat([x1_low, x2_low], 1)
        xc = self.DecCD(xc, xc_low)
        change = self.classifierCD(xc)

        return (F.interpolate(change, x_size[2:], mode='bilinear'),
                F.interpolate(out1, x_size[2:], mode='bilinear'),
                F.interpolate(out2,x_size[2:], mode='bilinear'))