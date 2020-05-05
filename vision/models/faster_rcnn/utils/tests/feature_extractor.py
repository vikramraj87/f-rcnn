from torchvision import models
from torch import nn


def decom_vgg16(use_drop=True):
    model = models.vgg16(pretrained=True)

    # Customize feature extractor
    features = list(model.features)[:30]

    # Freeze top 4 convolution layers
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    # Customize the classifier
    classifier = list(model.classifier)

    # Remove the final layer
    del classifier[6]

    if not use_drop:
        del classifier[5]
        del classifier[2]

    return nn.Sequential(*features), nn.Sequential(*classifier)