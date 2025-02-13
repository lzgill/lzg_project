'''
label align +  128 -> 8192*3
'''

from __future__ import print_function
import os
# import nni
import time
import torch
import logging
import argparse
import torchvision
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torch.nn import CTCLoss

from crnn_chs.my_pooling import my_MaxPool2d,my_AvgPool2d
import torchvision.transforms as transforms
character_vector_file = 'D:/dbnet-crnn/crnn_chs/character_vector_8192s.txt'
char_set_lines = open(character_vector_file, 'r', encoding='utf-8').readlines()
char_set = [ch.strip(' \n') for ch in char_set_lines]
assert 8192 == len(char_set)


def Mask(nb_batch, channels):
    foo = [1] * 2 + [0] * 1
    bar = []
    for i in range(8192):
        random.shuffle(foo)
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 8192 * channels, 1, 1)  ##bar  c:200 * channels

    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar

def supervisor(x, label, height, cnum):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    criterion = criterion.cuda()
    mask = Mask(x.size(0), cnum)
    branch = x
    branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
    branch = F.softmax(branch, 2)
    branch = branch.reshape(branch.size(0), branch.size(1), x.size(2), x.size(3))
    branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum  # set margin = 3.0


    branch_1 = x * mask
    branch_1 = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch_1)
    branch_1 = branch_1.squeeze(2)

    loss_1 = criterion(branch_1, label) # ctcloss

    return [loss_1, loss_2]


def CTC_loss(x, targets, length, ctc_loss):
    output = x.permute(2, 0, 1)  # [w, b, c] = [sl, bs, hs]
    output = output.contiguous()
    output = output.log_softmax(2)
    input_lengths = torch.full([output.size(1)], output.size(0), dtype=torch.long)
    target_lengths = length
    loss = ctc_loss(log_probs=output, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths)
    return loss


def encode_label(x):

    pred_list = x.permute(2, 0, 1)  # [w, b, c] = [sl, bs, hs]
    pred_list = pred_list.contiguous()
    pred_vectors = pred_list.argmax(axis=2)
    pred_vectors = pred_vectors.permute(1, 0)
    label = pred_vectors.cuda()
    return label


class CRNN(nn.Module):

    def __init__(self, channel=1, nclass=8192):
        super(CRNN, self).__init__()

        self.nclass = nclass
        self.fixed_width = False
        self.input_width = 480
        self.input_height = 32
        self.input_channel = channel

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
        self.conv7_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)

        self.bn2d7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.norm = nn.LayerNorm(128)
        self.fc8 = nn.Conv2d(in_channels=128, out_channels=self.nclass, kernel_size=1)

        #self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.LayerNorm(256)
        self.fc9 = nn.Conv2d(in_channels=256, out_channels=8192*3, kernel_size=1)
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

    def forward(self, inputs, targets=None, length=None, ctcloss=None):
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
        x = self.conv7(x)
        x = self.conv7_2(x)
        x = self.bn2d7(x)

        #-----------------
        #x1 = self.conv10(x)
        x1 = x.permute(0, 3, 2, 1)
        x1 = self.norm2(x1)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = self.fc9(x1)
        #-----------------

        x = self.conv8(x)    # 加一层
        x = self.conv9(x)
        x = x.permute(0, 3, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 2, 1) #x.shape :torch.Size([b, 128, 1, 120])
        x = self.fc8(x)   # [b, c, h, w], h=1  torch.Size([b, 8192, 1, 120])
        x = x.squeeze(2)  # [b, c, w]
        # x = x.permute(2, 0, 1)  # [w, b, c] equal to rnn's [h, b, c]
        # x = x.contiguous()
        label = encode_label(x)

        if self.training:
            assert targets is not None and length is not None and ctcloss is not None
            MC_loss = supervisor(x1, label, height=1, cnum=3)
            loss = CTC_loss(x, targets, length, ctcloss)
        if self.training:
            return x, loss, MC_loss
        else:
            return x





