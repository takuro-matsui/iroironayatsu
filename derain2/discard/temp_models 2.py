import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import numpy as np
from Util.util import Interpolate, UnNormfunc



class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, n_residual_blocks=16):
        super(GeneratorUNet, self).__init__() #おまじない


        nb_filter = [64, 128, 256, 512, 1024]

        self.conv_enc_0_1 = nn.Conv2d(in_channels, nb_filter[0], kernel_size=3, padding=1)
        self.bn_enc_0_1 = nn.BatchNorm2d(nb_filter[0])
        self.relu_enc_0_1 = nn.ReLU()

        self.enc_block0 = nn.Sequential(
                nn.Conv2d(in_channels, nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU()
            )

        self.pool0 = nn.MaxPool2d(2, 2)

        self.enc_block1 = nn.Sequential(
                nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
            )

        self.pool1 = nn.MaxPool2d(2, 2)


        self.enc_block2 = nn.Sequential(
                nn.Conv2d(nb_filter[1], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
            )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc_block3 = nn.Sequential(
                nn.Conv2d(nb_filter[2], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
            )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc_block4 = nn.Sequential(
                nn.Conv2d(nb_filter[3], nb_filter[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[4]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[4], nb_filter[4], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[4]),
                nn.ReLU(),
            )

        self.up4 = Interpolate(scale_factor=2, mode='bilinear')

        self.dec_block3 = nn.Sequential(
                nn.Conv2d(nb_filter[4]+nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[3], nb_filter[3], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[3]),
                nn.ReLU(),
            )

        self.up3 = Interpolate(scale_factor=2, mode='bilinear')

        self.dec_block2 = nn.Sequential(
                nn.Conv2d(nb_filter[3]+nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[2], nb_filter[2], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[2]),
                nn.ReLU(),
            )

        self.up2 = Interpolate(scale_factor=2, mode='bilinear')


        self.dec_block1 = nn.Sequential(
                nn.Conv2d(nb_filter[2]+nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[1]),
                nn.ReLU(),
            )

        self.up3 = Interpolate(scale_factor=2, mode='bilinear')

        self.dec_block0 = nn.Sequential(
                nn.Conv2d(nb_filter[1]+nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[0], nb_filter[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(nb_filter[0]),
                nn.ReLU(),
                nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            )



    
    def forward(self, x):
        
        x0_0 = self.enc_block0(x)
        x0_1 = self.pool0(x0_0)
        x1_0 = self.enc_block1(x0_1)
        x1_1 = self.pool1(x1_0)
        x2_0 = self.enc_block2(x1_1)
        x2_1 = self.pool2(x2_0)
        x3_0 = self.enc_block3(x2_1)
        x3_1 = self.pool3(x3_0)
        x4_0 = self.enc_block4(x3_1)
        x4_2 = torch.cat([self.up4(x4_0), x3_0], 1)
        x3_3 = self.dec_block3(x4_2)
        x3_2 = torch.cat([self.up4(x3_3), x2_0], 1)
        x2_3 = self.dec_block2(x3_2)
        x2_2 = torch.cat([self.up4(x2_3), x1_0], 1)
        x1_3 = self.dec_block1(x2_2)
        x1_2 = torch.cat([self.up4(x1_3), x0_0], 1)
        x0_3 = self.dec_block0(x1_2)

        #x_out = x * x0_3

        return x0_3


    


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__() #おまじない

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
