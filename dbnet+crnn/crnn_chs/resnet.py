import torch.nn as nn


class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = x * self.relu6(x + 3.0) / 6.0
        return out


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = act

        if self.act == 'relu':
            self.act_layer = nn.ReLU()
        elif self.act == 'hard_swish':
            self.act_layer = HSwish()
        elif self.act is None:
            self.act_layer = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act_layer(x)
        return x


class ConvBNPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, if_first=False):
        super().__init__()
        assert name is not None, 'shortcut must have name'

        self.name = name
        self.has_conv = False

        if in_channels != out_channels or stride[0] != 1:
            self.has_conv = True
            if if_first:
                self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      padding=0, groups=1, act=None)
            else:
                self.conv = ConvBNPool(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                       stride=stride, groups=1)
        elif if_first:
            self.has_conv = True
            self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                  padding=0, groups=1, act=None)
        else:
            self.conv = None

    def forward(self, x):
        if self.has_conv:
            x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'block must have name'
        self.name = name

        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                 name=f'{name}_branch1', if_first=if_first, )
        self.relu = nn.ReLU()
        self.output_channels = out_channels

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'bottleneck must have name'
        self.name = name
        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv2 = ConvBNACT(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1,
                               padding=0, groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * 4, stride=stride,
                                 if_first=if_first, name=f'{name}_branch1')
        self.relu = nn.ReLU()
        self.output_channels = out_channels * 4

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class ResNet(nn.Module):
    def __init__(self, in_channels, layers):
        super().__init__()
        supported_layers = {
            18: {'depth': [2, 2, 2, 2], 'block_class': BasicBlock},
            34: {'depth': [3, 4, 6, 3], 'block_class': BasicBlock},
            50: {'depth': [3, 4, 6, 3], 'block_class': BottleneckBlock},
            101: {'depth': [3, 4, 23, 3], 'block_class': BottleneckBlock},
            152: {'depth': [3, 8, 36, 3], 'block_class': BottleneckBlock},
            200: {'depth': [3, 12, 48, 3], 'block_class': BottleneckBlock}
        }
        assert layers in supported_layers, "supported layers {} but input layer is {}".format(supported_layers, layers)

        depth = supported_layers[layers]['depth']
        block_class = supported_layers[layers]['block_class']

        num_filters = [64, 128, 256, 512]
        self.conv1 = nn.Sequential(
            ConvBNACT(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, act='relu'),
            ConvBNACT(in_channels=32, out_channels=32, kernel_size=3, stride=1, act='relu', padding=1),
            ConvBNACT(in_channels=32, out_channels=64, kernel_size=3, stride=1, act='relu', padding=1)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        in_ch = 64
        for block_index in range(len(depth)):
            block_list = []
            for i in range(depth[block_index]):
                if layers >= 50:
                    if layers in [101, 152, 200] and block_index == 2:
                        if i == 0:
                            conv_name = "res" + str(block_index + 2) + "a"
                        else:
                            conv_name = "res" + str(block_index + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block_index + 2) + chr(97 + i)
                else:
                    conv_name = f'res{str(block_index + 2)}{chr(97 + i)}'
                if i == 0 and block_index != 0:
                    stride = (2, 1)
                else:
                    stride = (1, 1)
                block_list.append(block_class(in_channels=in_ch, out_channels=num_filters[block_index],
                                              stride=stride, if_first=block_index == i == 0, name=conv_name))
                in_ch = block_list[-1].output_channels
            self.stages.append(nn.Sequential(*block_list))
        self.out_channels = in_ch
        self.out = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        for stage in self.stages:
            x = stage(x)
        x = self.out(x)
        return x
