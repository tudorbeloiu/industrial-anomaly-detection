from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        ssim_val = ssim(pred, target, data_range=target.max() - target.min())
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * (1 - ssim_val)