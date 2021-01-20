###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Environmental Sound Classifier network for AI85/AI86
"""
import torch.nn as nn

import ai8x


class AI85ESC20Netv3(nn.Module):
    """
    Compound ESC20 v3 Audio net, all with Conv1Ds
    """

    # num_classes = n keywords + 1 unknown
    def __init__(
            self,
            num_classes=21,
            num_channels=128,
            dimensions=(128, 1),  # pylint: disable=unused-argument
            bias=False,
            **kwargs

    ):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        # Time: 128 Feature :128
        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, 100, 1, stride=1, padding=0,
                                                bias=bias, **kwargs)
        # T: 128 F: 100
        self.voice_conv2 = ai8x.FusedConv1dReLU(100, 96, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)
        # T: 126 F : 96
        self.voice_conv3 = ai8x.FusedMaxPoolConv1dReLU(96, 64, 3, stride=1, padding=1,
                                                       bias=bias, **kwargs)
        # T: 62 F : 64
        self.voice_conv4 = ai8x.FusedConv1dReLU(64, 48, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)
        # T : 60 F : 48
        self.esc_conv1 = ai8x.FusedMaxPoolConv1dReLU(48, 64, 3, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T: 30 F : 64
        self.esc_conv2 = ai8x.FusedConv1dReLU(64, 96, 3, stride=1, padding=0,
                                              bias=bias, **kwargs)
        # T: 28 F : 96
        self.esc_conv3 = ai8x.FusedAvgPoolConv1dReLU(96, 100, 3, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T : 14 F: 100
        self.esc_conv4 = ai8x.FusedMaxPoolConv1dReLU(100, 64, 6, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T : 2 F: 128
        self.fc = ai8x.Linear(256, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        x = self.drop(x)
        x = self.voice_conv3(x)
        x = self.voice_conv4(x)
        x = self.drop(x)
        x = self.esc_conv1(x)
        x = self.esc_conv2(x)
        x = self.drop(x)
        x = self.esc_conv3(x)
        x = self.esc_conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ai85esc20netv3(pretrained=False, **kwargs):
    """
    Constructs a AI85ESC20Net model.
    rn AI85ESC20Net(**kwargs)
    """
    assert not pretrained
    return AI85ESC20Netv3(**kwargs)


models = [
    {
        'name': 'ai85esc20netv3',
        'min_input': 1,
        'dim': 1,
    },
]
