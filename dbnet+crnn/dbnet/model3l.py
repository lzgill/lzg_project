import torch
import torch.nn as nn
from dbnet.mobile3l import MobileNetV3L
# from dbnet.ASF import FPN
# from dbnet.Bifpn import FPN
from dbnet.neck import FPN
from dbnet.head import DBHead


class DBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channel = 1

        self.backbone = MobileNetV3L(in_channels=self.input_channel, scale=0.5)
        self.neck = FPN(in_channels=self.backbone.out_channels, inner_channels=256)
        self.head = DBHead(in_channels=self.neck.out_channels)
        self.name = 'MobileNetV3L_FPN_DBHead'
        self.init_params()

    @classmethod
    def normalize(cls, inputs):
        inputs = inputs / 255.0
        inputs = inputs.transpose([0, 3, 1, 2])

        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inputs = (inputs - mean[None, :, None, None]) / std[None, :, None, None]
        inputs = torch.from_numpy(inputs).float()
        return inputs

    def init_params(self):
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

    def forward(self, inputs):
        _, _, h, w = inputs.size()
        backbone_out = self.backbone(inputs)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        return y
