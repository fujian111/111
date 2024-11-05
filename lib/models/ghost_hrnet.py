from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import math

import yaml
import torch.nn.functional as F


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, att_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_att = att_ratio is not None and att_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if att_ratio:
            self.att = CBAM(mid_chs)
        else:
            self.att = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.att is not None:
            x = self.att(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        # print("xxxxxxxx",x.shape)
        return x
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a=torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x = spatial_out * x
        # print('x:',x.shape)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthWiseConv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value
def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))
def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            # print("num_blocks[branch_index]",i)
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        # print('HR',len(x),self.num_branches)
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
            # print("x[i]",x[i].shape)

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse
blocks_dict = {
    'BASIC': BasicBlock,
    # 'BOTTLENECK': Bottleneck
}
class PoseHigherResolutionNet(nn.Module):
    def __init__(self,width = 1.0, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()
        # with open('D:/biye/hrnetoff/off_hrnet/experiments/coco/ghost_hrnet/frozen.yaml', 'r') as file:
        #     cfg = yaml.safe_load(file)
        with open('D:/biye/hrnetoff/off_hrnet/experiments/coco/ghost_hrnet/frozen_b1.yaml', 'r') as file:
            cfg = yaml.safe_load(file)
        # with open('D:/biye/hrnetoff/off_hrnet/experiments/mpii/ghost_hrnet/frozen.yaml', 'r') as file:
        #     cfg = yaml.safe_load(file)

        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel


        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        self.cfgs = [
            # k, t, c, att, s
            # stage1
            [[3, 16, 16, 0, 1]],
            # stage2
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            # stage3
            [[5, 72, 40, 0.25, 2]],
            [[5, 120, 40, 0.25, 1]],
            # stage4
            [[3, 240, 80, 0, 2]],
            [[3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
             ],
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
             ]
        ]
        for cfg11 in self.cfgs:

            layers = []
            for k, exp_size, c, att_ratio, s in cfg11:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    att_ratio=att_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        stages.append(nn.Sequential(ConvBnAct(input_channel, 320, 1)))

        self.blocks = nn.Sequential(*stages)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455, cfg['MODEL']['SCALE_FACTOR'])) * block.expansion)) for i in
            range(len(num_channels))
        ]
        if cfg['MODEL']['WIDTH_MULT'] == 1 and cfg['MODEL']['DEPTH_MULT'] == 1:
            self.trans1_branch1 = nn.Sequential(nn.Conv2d(24, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.trans1_branch2 = nn.Sequential(nn.Conv2d(40, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, cfg)  # should happen automatically

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        # print("self.stage3_cfg",self.stage3_cfg)
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455, cfg['MODEL']['SCALE_FACTOR'])) * block.expansion)) for i in
            range(len(num_channels))
        ]
        if cfg['MODEL']['WIDTH_MULT'] == 1 and cfg['MODEL']['DEPTH_MULT'] == 1:
            self.trans2_branch1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.trans2_branch2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.trans2_branch3 = nn.Sequential(nn.Conv2d(112, 128, 3, 1, 1), nn.BatchNorm2d(128),
                                                nn.ReLU(inplace=True))


        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, cfg)  # should happen automatically

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        # print("self.stage4_cfg",self.stage4_cfg)

        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455, cfg['MODEL']['SCALE_FACTOR'])) * block.expansion)) for i in
            range(len(num_channels))
        ]
        if cfg['MODEL']['WIDTH_MULT'] == 1 and cfg['MODEL']['DEPTH_MULT'] == 1:
            self.trans3_branch1 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.trans3_branch2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.trans3_branch3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128),
                                                nn.ReLU(inplace=True))
            self.trans3_branch4 = nn.Sequential(nn.Conv2d(320, 256, 3, 1, 1), nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True))


        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, cfg, multi_scale_output=False)  # should happen automatically
        # print("self.stage4",self.stage4)

        self.final_layers = self._make_final_layers(cfg, pre_stage_channels[0])



    def _make_final_layers(self, cfg, input_channels):
        dim_tag = cfg['MODEL']['NUM_JOINTS'] if cfg['MODEL']['TAG_PER_JOINT'] else 1
        extra = cfg['MODEL']['EXTRA']
        # scale_factor = 0
        final_layers = []
        # output_channels = cfg['MODEL']['NUM_JOINTS'] + dim_tag \
        #     if cfg['LOSS']['WITH_AE_LOSS'][0] else cfg['MODEL']['NUM_JOINTS']
        # print("1111",output_channels)
        output_channels = cfg['MODEL']['NUM_JOINTS']
        # print("cfg['MODEL']['NUM_JOINTS']",cfg['MODEL']['NUM_JOINTS'])
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        ))

        return nn.ModuleList(final_layers)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, cfg,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        # scale_factor = 0
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455, cfg['MODEL']['SCALE_FACTOR'])))) for i in
            range(len(num_channels))
        ]
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        for i in range(0, 3):
            x = self.blocks[i](x)  # 24*64*64
        x1 = x
        for i in range(3, 5):
            x = self.blocks[i](x)  # 40*32*32
        x2 = x
        for i in range(5, 7):
            x = self.blocks[i](x)  # 112*16*16
        x3 = x
        for i in range(7, 10):
            x = self.blocks[i](x)  # 320*8*8
        x4 = x

        x_list = []
        x_list.append(self.trans1_branch1(x1))  # 变通道32*64*64
        x_list.append(self.trans1_branch2(x2))#64*32*32
        y_list = self.stage2(x_list)

        x_list = []
        x_list.append(self.trans2_branch1(y_list[-2]))
        x_list.append(self.trans2_branch2(y_list[-1]))
        x_list.append(self.trans2_branch3(x3))
        y_list = self.stage3(x_list)

        x_list = []
        x_list.append(self.trans3_branch1(y_list[-3]))
        x_list.append(self.trans3_branch2(y_list[-2]))
        x_list.append(self.trans3_branch3(y_list[-1]))
        x_list.append(self.trans3_branch4(x4))

        y_list = self.stage4(x_list)

        final_outputs = []
        x = y_list[0]
        y = self.final_layers[0](x)
        final_outputs.append(y)
        return final_outputs[0]
def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet( **kwargs)

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #     model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model

