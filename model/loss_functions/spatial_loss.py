import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeansThreshold(nn.Module):
    def __init__(self):
        super(KMeansThreshold, self).__init__()

    def forward(self, x):
        b, c, p = x.size()
        x = x.contiguous().view(b * c, p)
        threshold, _ = torch.median(x, dim=1)

        while True:
            mask_low = x < threshold.unsqueeze(dim=1)
            mask_high = x >= threshold.unsqueeze(dim=1)
            l_center = torch.sum(x * mask_low, dim=1) / torch.count_nonzero(mask_low, dim=1).clamp(min=1)
            r_center = torch.sum(x * mask_high, dim=1) / torch.count_nonzero(mask_high, dim=1).clamp(min=1)
            n_threshold = (l_center + r_center) / 2
            if torch.equal(n_threshold, threshold):
                break
            else:
                threshold = n_threshold

        threshold = threshold.view(b, c)
        return threshold


class AdaptiveSpatialBCELoss(nn.Module):
    def __init__(self):
        super(AdaptiveSpatialBCELoss, self).__init__()
        self.eps = 1e-8
        self.region_len = 128
        self.threshold_mode = 1
        if self.threshold_mode == 1:
            self.f_kmeans = KMeansThreshold()
        self.loss_mode = 0
        self.include_neg = True
        self.criterion = nn.BCELoss()

    def forward(self, x, y):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x_sigmoid = torch.sigmoid(x)

        if self.threshold_mode == 0:
            threshold = x_sigmoid.view(b * c, h * w)
            threshold = torch.sort(threshold)[0]
            # 排序完成之后计算分布最稀疏段
            x_sigmoid_range = threshold[:, self.region_len - 1:] - threshold[:,
                                                                   :-self.region_len + 1]  # 计算所有长为region_len段的极差
            _, x_sigmoid_range = torch.max(x_sigmoid_range, dim=1)  # 获得极值最大值所在下标
            x_sigmoid_range = x_sigmoid_range.int()
            x_sigmoid_range = x_sigmoid_range + self.region_len // 2  # 获得极值最大区间中位数所在下标 x_sigmoid_range:[bc]

            temp_mask = torch.zeros_like(threshold)
            for i in range(b * c):
                temp_mask[i][x_sigmoid_range[i]] = 1
            threshold = torch.masked_select(threshold, temp_mask.bool())
            threshold = threshold.view(b, c)

        elif self.threshold_mode == 1:
            threshold = self.f_kmeans(x_sigmoid)  # !!REVISE!!: change x to x_sigmoid

        else:
            raise NotImplementedError()

        threshold = torch.stack([threshold for _ in range(h * w)], dim=2)
        threshold = threshold.detach()

        threshold = threshold.clamp(min=0.0001)
        y_ = torch.stack([y for _ in range(h * w)], dim=2)  # y_:[b,c,hw]
        mask_low = (x_sigmoid <= threshold) * y_
        mask_high = (x_sigmoid > threshold) * y_
        mask_low = mask_low.detach()
        mask_high = mask_high.detach()

        if self.loss_mode == 0:
            h_low = (-(torch.pow(x_sigmoid, 2) / torch.pow(threshold, 2)) + 2 * x_sigmoid / threshold) * mask_low
            alpha = 1 / torch.pow(1 - threshold, 2).clamp(min=self.eps)
            h_high = (alpha * (1 - x_sigmoid) * (1 - 2 * threshold + x_sigmoid)) * mask_high

            piecewise_spatial_bceloss = h_high + h_low
            if self.include_neg:
                neg_loss = -(1 - y_) * torch.log((1 - x_sigmoid).clamp(min=self.eps))
                piecewise_spatial_bceloss = piecewise_spatial_bceloss + neg_loss
                piecewise_spatial_bceloss = torch.mean(piecewise_spatial_bceloss)
            else:
                piecewise_spatial_bceloss = torch.sum(piecewise_spatial_bceloss) / torch.sum(y_)

        elif self.loss_mode == 1:
            piecewise_spatial_bceloss = self.criterion(x_sigmoid, mask_high)

        else:
            piecewise_spatial_bceloss = None

        return piecewise_spatial_bceloss
