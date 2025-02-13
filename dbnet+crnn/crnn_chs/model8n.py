from torch import nn
import torch


class CRNN(nn.Module):

    def __init__(self, channel=1, nclass=8192):
        super(CRNN, self).__init__()

        self.nclass = nclass
        self.fixed_width = False
        #
        self.input_width = 480
        self.input_height = 32
        self.input_channel = channel
        # kernel_size=3, padding=1 no change the size of feature maps
        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=3, padding=1) # yq 修改
        self.bn2d1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2d4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2d5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.relu6 = nn.ReLU(inplace=True)
        self.pool6_l = nn.MaxPool2d(kernel_size=(2, 1))
        self.pool6_r = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(2, 1), padding=0, bias=False)
        # self.conv7_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)

        self.bn2d7 = nn.BatchNorm2d(128)
        # rnn
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.norm = nn.LayerNorm(128)

        self.fc8 = nn.Conv2d(in_channels=128, out_channels=self.nclass, kernel_size=1)

        self.init_params()

    def init_params(self, method='xavier'):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    @classmethod
    def normalize(cls, inputs):
        inputs = inputs / 255.0
        inputs = inputs.transpose([0, 3, 1, 2])
        inputs = torch.from_numpy(inputs).float()

        return inputs

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn2d1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2d2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn2d3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn2d4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn2d5(x)
        x = self.relu5(x)

        pl = self.conv6(x)
        pl = self.relu6(pl)
        pl = self.pool6_l(pl)
        pr = self.pool6_r(x)
        x = torch.cat([pl, pr], dim=1)
        # print(x.shape)
        x = self.conv7(x)
        # x=self.conv7_2(x)

        x = self.bn2d7(x)
        x = self.conv8(x)   # 加一层
        x = self.conv9(x)

        x = x.permute(0, 3, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 2, 1)  # x.shape :torch.Size([b, 128, 1, 120])

        x = self.fc8(x)   # [b, c, h, w], h=1  torch.Size([b, 8192, 1, 120])

        x = x.squeeze(2)  # [b, c, w]
        # x = x.permute(2, 0, 1)  # [w, b, c] equal to rnn's [h, b, c]
        # x = x.contiguous()
        return x
