import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskConsistencyLoss(nn.Module):
    def __init__(self, num_masks=255):
        super(MaskConsistencyLoss, self).__init__()
        self.eps = 1e-8
        self.num_masks = num_masks
        self.criterion = nn.MSELoss()

    def forward(self, x, mask=None, ignore_bg=True):
        b, c, h, w = x.size()
        mask = F.interpolate(mask, size=[h, w], mode='nearest')
        x = x.view(b, c, h * w)
        x_sigmoid = torch.sigmoid(x)
        mask = mask.view(b, h * w)

        label = torch.zeros(b, c, h * w).to(x.device)
        for i in range(self.num_masks):  # mask:[b,h*w]
            cur_mask = (mask == i)
            cur_pix = torch.count_nonzero(cur_mask, dim=1)  # cur_pix:[b]
            x_masked = x_sigmoid * (cur_mask.unsqueeze(dim=1))
            # x_sigmoid:[b,c,h*w], cur_mask:[b,h*w], x_masked:[b,c,h*w]
            cur_sum = torch.sum(x_masked, dim=2)  # cur_sum:[b,c]
            cur_avg = cur_sum / (torch.where(cur_pix == 0, torch.ones_like(cur_pix), cur_pix).unsqueeze(dim=1))
            # cur_avg:[b,c]
            label = torch.where(cur_mask.unsqueeze(dim=1).expand(b, c, h * w),
                                cur_avg.unsqueeze(dim=2).expand(b, c, h * w), label)

        if ignore_bg:
            x_sigmoid = torch.where((mask == 255).unsqueeze(dim=1).expand(b, c, h * w), torch.zeros_like(x_sigmoid),
                                    x_sigmoid)

        loss = self.criterion(x_sigmoid, label)

        return loss
