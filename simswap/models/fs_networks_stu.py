"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, 2)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.GAP(self.conv1(x)).view(-1, 128)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.fc1(x)
        return x

class WMEbeddingNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(WMEbeddingNet, self).__init__()

        self.base_net = nn.Sequential(
                nn.Linear(in_planes, 128),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Linear(128, out_planes))

    def forward(self, x):
        x = self.base_net(x)
        return x

class CustomConvLayer(nn.Module):
    def __init__(self, in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1):  #256x256x64 
        super(CustomConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.transformation = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size), # 112
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size, stride=2), # 56
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size, stride=2), # 28
            nn.LeakyReLU(inplace=True)
        )

        self.t1 = nn.Conv2d(64, 128, kernel_size, stride=2) # 14
        self.t2 = nn.Conv2d(64, 128, kernel_size, stride=2)
        self.t3 = nn.Conv2d(64, 128, kernel_size, stride=2)
        self.t4 = nn.Conv2d(64, 128, kernel_size, stride=2)

        # self.wm_embed = WMEbeddingNet(32, 64)
        self.attention = Attention()
        self.weight = nn.Parameter(torch.randn(1, 4, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), requires_grad=True) # 1 x K x 64 x 3 x 3 x 3
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.wm_embed = WMEbeddingNet(32, 128)


    def forward(self, x, wm):  # wm_coff: B x 256
        batch_size, in_planes, height, width = x.size()   # B x 64 x 256 x 256  -> B x 3 x 64 x (3 x 3)

        wm_coff = self.wm_embed(wm).view(batch_size, -1, 1, 1)
        x = wm_coff * x
        
        x_hat = F.avg_pool2d(x, kernel_size=2, stride=2)  #  B x 64 x 128 x 128
        transformed_weights = self.transformation(x_hat)  #  B x 64 x 16 x 16
        transformed_weight_1 = self.GAP(self.t1(transformed_weights)).view(batch_size, 1, 1, -1, 1, 1)
        transformed_weight_2 = self.GAP(self.t2(transformed_weights)).view(batch_size, 1, 1, -1, 1, 1)
        transformed_weight_3 = self.GAP(self.t3(transformed_weights)).view(batch_size, 1, 1, -1, 1, 1)
        transformed_weight_4 = self.GAP(self.t4(transformed_weights)).view(batch_size, 1, 1, -1, 1, 1) #  B x 64 x 1 x 1

        aggregate_coff = torch.concat((transformed_weight_1, transformed_weight_2, transformed_weight_3, transformed_weight_4), dim=1) # 32 x 4 x 64 x 64 x 1
        aggregate_weightv1 = aggregate_coff * self.weight

        kernel_attention = self.attention(transformed_weights).view(batch_size, -1, 1, 1, 1, 1)
        aggregate_weightv2 = kernel_attention * aggregate_weightv1     # 1 x K x 64 x 3 x 3 x 3

        aggregate_weightv3 = torch.sum(aggregate_weightv2, dim=1).view([-1, in_planes, self.kernel_size, self.kernel_size])

        x = x.view(1, -1, height, width)
        output = F.conv2d(x, weight = aggregate_weightv3, bias=None, stride=self.stride, padding=self.padding, groups=batch_size)
        output = output.view(batch_size, self.out_channels , output.size(-2), output.size(-1))

        return output




class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out



class Generator_Adain_Upsample_stu(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(Generator_Adain_Upsample_stu, self).__init__()
        activation = nn.ReLU(True)
        self.deep = deep

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
        if self.deep:
            self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                       norm_layer(512), activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        # self.up1 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64), activation
        # )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn = nn.BatchNorm2d(64)
        self.activation = activation

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
                                        nn.Tanh())
        
        self.id_conv = CustomConvLayer(128, 64, 3, 1, 1) #128->64  224


    def forward(self, input, dlatents, wm):
        x = input  # 3*224*224

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)

        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)

        if self.deep:
            x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        # x = self.up1(x)

        x = self.upsample(x)
        x = self.id_conv(x, wm)
        x = self.bn(x)
        x = self.activation(x)

        x = self.last_layer(x)
        x = (x + 1) / 2

        return x

class Discriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kw, stride=2, padding=padw),
            norm_layer(128), nn.LeakyReLU(0.2, True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kw, stride=2, padding=padw),
            norm_layer(256), nn.LeakyReLU(0.2, True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kw, stride=2, padding=padw),
            norm_layer(512), nn.LeakyReLU(0.2, True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kw, stride=1, padding=padw),
            norm_layer(512),
            nn.LeakyReLU(0.2, True)
        )

        if use_sigmoid:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
            )

    def forward(self, input):
        out = []
        x = self.down1(input)
        out.append(x)
        x = self.down2(x)
        out.append(x)
        x = self.down3(x)
        out.append(x)
        x = self.down4(x)
        out.append(x)
        x = self.conv1(x)
        out.append(x)
        x = self.conv2(x)
        out.append(x)
        
        return out