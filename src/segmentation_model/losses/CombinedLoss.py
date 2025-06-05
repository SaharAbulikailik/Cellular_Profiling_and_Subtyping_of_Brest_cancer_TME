import torch
import torch.nn as nn
from monai.losses import DiceLoss
from losses.CurvatureLoss import CurvatureKLDivergenceLoss

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, curvature_weight=0.5, sigma=4, pdf_bins=1000):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.curvature_loss = CurvatureKLDivergenceLoss(sigma=sigma, pdf_bins=pdf_bins)
        self.dice_weight = dice_weight
        self.curvature_weight = curvature_weight

    def forward(self, y_pred, y_true):
        dice_loss_value = self.dice_loss(y_pred, y_true)
        curvature_loss_value = self.curvature_loss(y_pred, y_true)
        total_loss = self.dice_weight * dice_loss_value + self.curvature_weight * curvature_loss_value
        return total_loss
