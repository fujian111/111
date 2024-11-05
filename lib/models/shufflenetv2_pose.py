from typing import List, Callable
# from mmcv.cnn import (build_conv_layer,  build_upsample_layer)
import torch
from torch import Tensor
import torch.nn as nn
# from torchsummary import summary

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x
def _make_deconv_layer(in_channels,num_layers, num_filters, num_kernels):
    """Make deconv layers."""
    if num_layers != len(num_filters):
        error_msg = f'num_layers({num_layers}) ' \
                    f'!= length of num_filters({len(num_filters)})'
        raise ValueError(error_msg)
    if num_layers != len(num_kernels):
        error_msg = f'num_layers({num_layers}) ' \
                    f'!= length of num_kernels({len(num_kernels)})'
        raise ValueError(error_msg)

    layers = []
    for i in range(num_layers):
        kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i])

        planes = num_filters[i]
        #mmpose的反卷积
        # layers.append(
        #     build_upsample_layer(
        #         dict(type='deconv'),
        #         in_channels=in_channels,
        #         out_channels=planes,
        #         kernel_size=kernel,
        #         stride=2,
        #         padding=padding,
        #         output_padding=output_padding,
        #         bias=False))
        layers.append(nn.ConvTranspose2d(in_channels,
                                         planes,
                                         kernel,
                                         stride=2,
                                         padding=padding,
                                         output_padding=output_padding,
                                         bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        in_channels = planes

    return nn.Sequential(*layers)
def _get_deconv_cfg(deconv_kernel):
    """Get configurations for deconv layers."""
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
        return deconv_kernel,padding,output_padding
    else:
        raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 cfg,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
       #反卷积层
        num_deconv_layers = 3
        if num_deconv_layers > 0:
            self.deconv_layers = _make_deconv_layer(
                in_channels=1024,
                num_layers=3,
                num_filters=(256, 256, 256),
                num_kernels=(4, 4, 4)
            )
        #全连接层(mmpose)
        # self.final_layer = build_conv_layer(
        #     cfg=dict(type='Conv2d'),
        #     in_channels=256,
        #     out_channels=17,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0)
        # 全连接层
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=1,
            stride=1,
            padding=0
        )
        # self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #shufflenetv2
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        #反卷积
        x = self.deconv_layers(x)
        #全连接层
        x = self.final_layer(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def shufflenet_v2_x1_0(cfg):

    model = ShuffleNetV2(cfg,
                         stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         )

    return model
def get_pose_net(cfg, is_train, **kwargs):
    model = shufflenet_v2_x1_0(cfg)
    return model
#off_hrnet:只计算卷积层和全连接层，没计算反卷积层，而且不计算偏置
#方法三【torchstat】跟dite_hrnet【torchstat1】方法本质是一样的，
# 但是【torchstat】的GFlops=flops/1000/1000/1000;而【torchstat1】的GFlops=flops/1024/1024/1024
# print("##############方法三：torchstat 计算骨干网络#####################")
# from torchstat import stat
# model=shufflenet_v2_x1_0()
# stat(model, (3, 256, 256))
#
# print("####################dite_hrnet作者的方法#torchstat1######计算骨干网络##########################")
# from torchstat_utils import model_stats
# model_stats(model, (1,3, 256, 256))