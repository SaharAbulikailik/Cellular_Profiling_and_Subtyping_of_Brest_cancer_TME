import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from skimage import measure

class CurvatureKLDivergenceLoss(nn.Module):
    def __init__(self, sigma=4, pdf_bins=1000):
        super(CurvatureKLDivergenceLoss, self).__init__()
        self.sigma = sigma
        self.pdf_bins = pdf_bins
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

    def compute_curvature_gaussian(self, x, y):
        dx = gaussian_filter1d(x, sigma=self.sigma, order=1, truncate=4.5)
        dy = gaussian_filter1d(y, sigma=self.sigma, order=1, truncate=4.5)
        ddx = gaussian_filter1d(dx, sigma=self.sigma, order=1, truncate=4.5)
        ddy = gaussian_filter1d(dy, sigma=self.sigma, order=1, truncate=4.5)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
        return curvature

    def calculate_curvature_data(self, image):
        if image.ndim == 4:
            image = image[0, 0]
        elif image.ndim == 3:
            image = image[0]

        contours_data = []
        labeled_image, _ = measure.label(image, return_num=True)
        for region in measure.regionprops(labeled_image, intensity_image=image):
            mask = (labeled_image == region.label)
            contours = measure.find_contours(mask)
            for contour in contours:
                X_p = contour[:, 1]
                Y_p = contour[:, 0]
                curvature_data = self.compute_curvature_gaussian(X_p, Y_p)
                contours_data.append(curvature_data)
        return np.concatenate(contours_data) if contours_data else np.array([0])

    def forward(self, y_pred, y_true):
        y_pred = (y_pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
        y_true = y_true.detach().cpu().numpy().astype(np.uint8)

        curvature_pred = self.calculate_curvature_data(y_pred)
        curvature_true = self.calculate_curvature_data(y_true)

        if len(curvature_pred) < 2 or len(curvature_true) < 2:
            return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        pdf_pred = gaussian_kde(curvature_pred)
        pdf_true = gaussian_kde(curvature_true)

        x_vals = np.linspace(min(curvature_true.min(), curvature_pred.min()),
                             max(curvature_true.max(), curvature_pred.max()), self.pdf_bins)
        pdf_vals_pred = pdf_pred(x_vals) + 1e-10
        pdf_vals_true = pdf_true(x_vals) + 1e-10

        pdf_vals_pred = torch.tensor(pdf_vals_pred, dtype=torch.float32).log()
        pdf_vals_true = torch.tensor(pdf_vals_true, dtype=torch.float32)

        kl_div = self.kldiv_loss(pdf_vals_pred, pdf_vals_true)
        return torch.tensor(kl_div, dtype=torch.float32, requires_grad=True)
