import torch
import torch.nn as nn
import torch.nn.functional as F


class ForegroundRatioLoss(nn.Module):
    def __init__(self):
        super(ForegroundRatioLoss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y, fg):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        prob = torch.stack([xs_neg, xs_pos], dim=2)
        marginal = torch.mean(prob, dim=3)

        y_ = torch.stack([y, y], dim=2)

        bg = 1 - fg
        fb = torch.stack([bg, fg], dim=2)

        # kl_loss = (y_ * marginal * torch.log((marginal / (fb + 1e-10)).clamp(min=self.eps))).sum(2)
        kl_loss = (y_ * fb * (fb.clamp(min=self.eps) / marginal.clamp(min=self.eps)).log()).sum(2)
        kl_loss = torch.sum(kl_loss) / torch.sum(y)

        return kl_loss