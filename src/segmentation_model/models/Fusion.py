import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Channel Attention Module
# --------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

# --------------------------
# Spatial Attention Module
# --------------------------
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([max_out, avg_out], dim=1)
        attention = self.sigmoid(self.bn(self.conv(x_cat)))
        return x * attention

# --------------------------
# CBAM: Channel + Spatial Attention
# --------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

# --------------------------
# CBAM Fusion Block
# --------------------------
class CBAMFusionBlock(nn.Module):
    def __init__(self, ch1, ch2, ch_out):
        super(CBAMFusionBlock, self).__init__()
        self.cbam1 = CBAM(ch1)
        self.cbam2 = CBAM(ch2)

        self.fusion = nn.Sequential(
            nn.Conv2d(ch1 + ch2, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.cbam1(x1)
        x2 = self.cbam2(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.fusion(x)
