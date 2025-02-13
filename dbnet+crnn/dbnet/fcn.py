
import torch.nn as nn


class FCN5(nn.Module):

    def __init__(self, nc):
        super(FCN5, self).__init__()
        # 1 / 2
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 4
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 8
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 16
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2d4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 32
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2d5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_channels = [64, 128, 128, 128]

    def forward(self, inputs):
        x = inputs

        out = []

        x = self.conv1(x)
        x = self.bn2d1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # out.append(x)

        x = self.conv2(x)
        x = self.bn2d2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        out.append(x)

        x = self.conv3(x)
        x = self.bn2d3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        out.append(x)

        x = self.conv4(x)
        x = self.bn2d4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        out.append(x)

        x = self.conv5(x)
        x = self.bn2d5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        out.append(x)

        return out


class FCN10(nn.Module):

    def __init__(self, nc):
        super(FCN10, self).__init__()
        # 1 / 2
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 4
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2d3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 8
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2d5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 16
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2d7 = nn.BatchNorm2d(128)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1 / 32
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn2d9 = nn.BatchNorm2d(128)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)
        self.pool10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_channels = [64, 128, 128, 128]

    def forward(self, inputs):
        x = inputs

        out = []

        x = self.conv1(x)
        x = self.bn2d1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # out.append(x)

        x = self.conv3(x)
        x = self.bn2d3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        out.append(x)

        x = self.conv5(x)
        x = self.bn2d5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        out.append(x)

        x = self.conv7(x)
        x = self.bn2d7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.pool8(x)
        out.append(x)

        x = self.conv9(x)
        x = self.bn2d9(x)
        x = self.relu9(x)

        x = self.conv10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        out.append(x)

        return out
