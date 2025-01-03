import torch
import torch.nn as nn
from monai.losses import DiceLoss

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.25, epsilon=0.8):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')(y_pred, y_true)
        num_loss = self.calculate_num_loss(y_pred, y_true)
        cell_loss = self.calculate_cellularity_loss(y_pred, y_true)
        return self.alpha * dice_loss + self.beta * num_loss + self.epsilon * cell_loss

    def calculate_num_loss(self, y_pred, y_true):
        y_pred_bin = y_pred > 0.5
        y_true_bin = y_true > 0.5
        num_pred = y_pred_bin.sum().item()
        num_true = y_true_bin.sum().item()
        return abs(num_pred - num_true) / (num_true + 1e-6)

    def calculate_cellularity_loss(self, y_pred, y_true):
        y_pred_bin = y_pred > 0.5
        y_true_bin = y_true > 0.5
        area_pred = y_pred_bin.sum().item()
        area_true = y_true_bin.sum().item()
        return abs(area_pred - area_true) / (area_true + 1e-6)
