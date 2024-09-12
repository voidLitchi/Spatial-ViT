import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.backbone as encoder


class BaseNet(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") \
            else [self.encoder.embed_dims[-1]] * 4

        self.middle_layer_channels = 128

        self.mask_layer_downsample = nn.Linear(in_features=self.in_channels[-1],
                                               out_features=self.middle_layer_channels, bias=False)
        torch.nn.init.xavier_uniform_(self.mask_layer_downsample.weight)

        self.classifier = nn.Linear(in_features=self.middle_layer_channels,
                                    out_features=self.num_classes - 1, bias=False)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        cls_token, x = self.encoder.forward_features(x)
        x = F.relu(x)
        label_feature = []
        x = self.mask_layer_downsample(x)
        x = self.classifier(x)
        b, hw, c = x.shape
        h = w = math.isqrt(hw)
        x = x.transpose(1, 2).reshape(b, c, h, w)

        return x

    def get_param_groups(self):

        param_groups = [[], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.mask_layer_downsample.weight)
        param_groups[2].append(self.classifier.weight)

        return param_groups


def main():
    model = BaseNet('deit_base_patch16_224', 21, True, 0.9, -3)
    x = torch.rand(4, 3, 448, 448)
    predict = model.forward(x)
    print(predict)


if __name__ == '__main__':
    main()
