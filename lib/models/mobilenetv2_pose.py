from torch import nn
import torch
# from torchsummary import summary


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch
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
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, cfg, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        num_deconv_layers = 3
        # num_deconv_filters = (256, 256, 256)
        # num_deconv_kernels = (4, 4, 4)
        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            print("output_channel",output_channel)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        #特征提取部分
        self.features = nn.Sequential(*features)
        #反卷积
        if num_deconv_layers > 0:
            self.deconv_layers = _make_deconv_layer(
                in_channels=1280,
                num_layers=3,
                num_filters=(256, 256, 256),
                num_kernels=(4, 4, 4)
            )

        #全连接层（mmpose）
        # self.final_layer=build_conv_layer(
        #     cfg=dict(type='Conv2d'),
        #     in_channels=256,
        #     out_channels=17,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0)
        #全连接层
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=1,
            stride=1,
            padding=0
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

def get_pose_net(cfg, is_train, **kwargs):
    model = MobileNetV2(cfg)
    return model
#off_hrnet:只计算卷积层和全连接层，没计算反卷积层，而且不计算偏置
#方法三【torchstat】跟dite_hrnet【torchstat1】方法本质是一样的，
# 但是【torchstat】的GFlops=flops/1000/1000/1000;而【torchstat1】的GFlops=flops/1024/1024/1024
# print("##############方法三：torchstat 计算骨干网络#####################")
# from torchstat import stat
# model=MobileNetV2(cfg)
# stat(model, (3, 256, 256))
#
# print("####################dite_hrnet作者的方法#torchstat1######计算骨干网络##########################")
# from torchstat_utils import model_stats
# model_stats(model, (1,3, 256, 256))