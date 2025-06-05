import torch
import torch.nn as nn
from models.DoubleConv import DoubleConv, DecoderBlock
from models.Fusion import CBAMFusionBlock
from models.swin_T_1 import SwinTransformer

class SwinTransformerVit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinTransformerVit, self).__init__()
        self.swinViT = SwinTransformer()

        self.encoder1 = DoubleConv(in_channels, 48)
        self.encoder2 = DoubleConv(48, 96)
        self.encoder3 = DoubleConv(96, 192)
        self.encoder4 = DoubleConv(192, 384)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)

        self.conv_in_img = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.conv1_img = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv2_img = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.conv3_img = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv4_img = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))

        self.fuse1 = CBAMFusionBlock(48, 48, 96)
        self.fuse2 = CBAMFusionBlock(96, 96, 192)
        self.fuse3 = CBAMFusionBlock(192, 192, 384)
        self.fuse4 = CBAMFusionBlock(384, 384, 768)

        self.decoder1 = DecoderBlock(768, 384)
        self.decoder2 = DecoderBlock(384, 192)
        self.decoder3 = DecoderBlock(192, 96)
        self.decoder4 = DecoderBlock(96, 48)

        self.final_conv = nn.Conv2d(48, out_channels, kernel_size=1)

    def forward(self, x, saliency):
        skip0 = self.conv_in_img(x)
        skip1, skip2, skip3, skip4 = self.swinViT(x)
        skip1 = self.conv1_img(skip1)
        skip2 = self.conv2_img(skip2)
        skip3 = self.conv3_img(skip3)
        skip4 = self.conv4_img(skip4)

        x_sal = self.encoder1(saliency)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip1_sal = x_sal

        x_sal = self.encoder2(x_sal)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip2_sal = x_sal

        x_sal = self.encoder3(x_sal)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip3_sal = x_sal

        x_sal = self.encoder4(x_sal)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip4_sal = x_sal

        F_Skip1 = self.fuse1(skip1, skip1_sal)
        F_Skip2 = self.fuse2(skip2, skip2_sal)
        F_Skip3 = self.fuse3(skip3, skip3_sal)
        F_Skip4 = self.fuse4(skip4, skip4_sal)

        x = self.decoder1(F_Skip4, F_Skip3)
        x = self.decoder2(x, F_Skip2)
        x = self.decoder3(x, F_Skip1)
        x = self.decoder4(x, skip0)
        out = self.final_conv(x)
        return out
