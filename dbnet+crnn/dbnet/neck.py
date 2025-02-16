import torch
import torch.nn as nn
import torch.nn.functional


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256):
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4

        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )
        self.out_channels = self.conv_out

    def forward(self, x):
        c2, c3, c4, c5 = x

        p5 = self.reduce_conv_c5(c5)

        p4 = self.reduce_conv_c4(c4) + torch.nn.functional.interpolate(p5, scale_factor=2)
        p4 = self.smooth_p4(p4)

        p3 = self.reduce_conv_c3(c3) + torch.nn.functional.interpolate(p4, scale_factor=2)
        p3 = self.smooth_p3(p3)

        p2 = self.reduce_conv_c2(c2) + torch.nn.functional.interpolate(p3, scale_factor=2)
        p2 = self.smooth_p2(p2)
        # interpolate, scale_factor > 0 上采样, scale_factor < 0 下采样
        p5 = torch.nn.functional.interpolate(p5, scale_factor=8)
        p4 = torch.nn.functional.interpolate(p4, scale_factor=4)
        p3 = torch.nn.functional.interpolate(p3, scale_factor=2)

        x = torch.cat([p2, p3, p4, p5], dim=1)

        x = self.conv(x)

        return x


class ASF(nn.Module):
    def __init__(self, in_channels, inner_channels=256):
        super().__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4

        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1)
        self.conv_re_conv_sig = nn.Sequential(
            nn.Conv2d(1, self.conv_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.conv_out, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.conv_out, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.out_channels = self.conv_out

    def forward(self, x):
        c2, c3, c4, c5 = x

        p5 = self.reduce_conv_c5(c5)

        p4 = self.reduce_conv_c4(c4) + torch.nn.functional.interpolate(p5, scale_factor=2)
        p4 = self.smooth_p4(p4)

        p3 = self.reduce_conv_c3(c3) + torch.nn.functional.interpolate(p4, scale_factor=2)
        p3 = self.smooth_p3(p3)

        p2 = self.reduce_conv_c2(c2) + torch.nn.functional.interpolate(p3, scale_factor=2)
        p2 = self.smooth_p2(p2)
        # interpolate, scale_factor > 0 上采样, scale_factor < 0 下采样
        p5 = torch.nn.functional.interpolate(p5, scale_factor=8)
        p4 = torch.nn.functional.interpolate(p4, scale_factor=4)
        p3 = torch.nn.functional.interpolate(p3, scale_factor=2)
        #
        x = torch.cat([p2, p3, p4, p5], dim=1)   # 维度相加

        x1 = self.conv(x)
        x_sa = torch.mean(x1, dim=1, keepdim=True)
        x_sa = self.conv_re_conv_sig(x_sa)
        x_attention = x_sa + x1
        x_attention = self.conv1(x_attention)
        x = x * x_attention
        return x
