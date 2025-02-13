import torch.nn as nn


class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = x * self.relu6(x + 3.0) / 6.0
        return out


class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.relu6(x + 3.0) / 6.0
        return out


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'hard_swish':
            self.act = HSwish()
        elif act is None:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super().__init__()
        num_mid_filter = out_channels // ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_mid_filter, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_mid_filter, kernel_size=1, out_channels=out_channels, bias=True)
        self.relu2 = HardSigmoid()

    def forward(self, x):
        attn = self.pool(x)
        attn = self.conv1(attn)
        attn = self.relu1(attn)
        attn = self.conv2(attn)
        attn = self.relu2(attn)
        return x * attn


class ResidualUnit(nn.Module):
    def __init__(self, num_in_filter, num_mid_filter, num_out_filter, stride, kernel_size, act=None, use_se=False):
        super().__init__()
        self.conv0 = ConvBNACT(in_channels=num_in_filter, out_channels=num_mid_filter, kernel_size=1, stride=1,
                               padding=0, act=act)

        self.conv1 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_mid_filter, kernel_size=kernel_size,
                               stride=stride,
                               padding=int((kernel_size - 1) // 2), act=act, groups=num_mid_filter)
        if use_se:
            self.se = SEBlock(in_channels=num_mid_filter, out_channels=num_mid_filter)
        else:
            self.se = None

        self.conv2 = ConvBNACT(in_channels=num_mid_filter, out_channels=num_out_filter, kernel_size=1, stride=1,
                               padding=0)
        self.shortcut = False
        if num_in_filter == num_out_filter and stride == 1:
            self.shortcut = True

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y)

        if self.shortcut:
            y = x + y
        return y


class MobileNetV3S(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.scale = 0.5
        self.inplanes = 16

        self.cfg = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, True, 'relu', 2],
            [3, 72, 24, False, 'relu', 2],
            [3, 88, 24, False, 'relu', 1],
            [5, 96, 40, True, 'hard_swish', 2],
            [5, 240, 40, True, 'hard_swish', 1],
            [5, 240, 40, True, 'hard_swish', 1],
            [5, 120, 48, True, 'hard_swish', 1],
            [5, 144, 48, True, 'hard_swish', 1],
            [5, 288, 96, True, 'hard_swish', 2],
            [5, 576, 96, True, 'hard_swish', 1],
            [5, 576, 96, True, 'hard_swish', 1],
        ]
        self.cls_ch_squeeze = 576
        self.cls_ch_expand = 1280

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert self.scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, self.scale)

        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        # conv1
        self.conv1 = ConvBNACT(in_channels=in_channels,
                               out_channels=self.make_divisible(inplanes * scale),
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               groups=1,
                               act='hard_swish')
        i = 0
        inplanes = self.make_divisible(inplanes * scale)
        self.stages = nn.ModuleList()

        block_list = []
        self.out_channels = []
        for layer_cfg in cfg:
            if layer_cfg[5] == 2 and i > 0:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block = ResidualUnit(num_in_filter=inplanes,
                                 num_mid_filter=self.make_divisible(scale * layer_cfg[1]),
                                 num_out_filter=self.make_divisible(scale * layer_cfg[2]),
                                 act=layer_cfg[4],
                                 stride=layer_cfg[5],
                                 kernel_size=layer_cfg[0],
                                 use_se=layer_cfg[3])
            block_list.append(block)
            inplanes = self.make_divisible(scale * layer_cfg[2])
            i += 1
        self.stages.append(nn.Sequential(*block_list))
        self.conv2 = ConvBNACT(
            in_channels=inplanes,
            out_channels=self.make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act='hard_swish')
        self.out_channels.append(self.make_divisible(scale * cls_ch_squeeze))

    @classmethod
    def make_divisible(cls, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        x = self.conv1(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        out[-1] = self.conv2(out[-1])
        return out
