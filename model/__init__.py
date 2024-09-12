from .base import BaseNet


def get_model(backbone, num_classes=None, pretrained=None, init_momentum=None):
    model = BaseNet(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        init_momentum=init_momentum
    )
    return model
