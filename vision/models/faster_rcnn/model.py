from torch import nn
from vision.models.faster_rcnn.utils.tests.feature_extractor import decom_vgg16


class Model(nn.Module):
    # Down sample 16x from input to conv5
    feat_stride = 16

    def __init__(self,
                 n_fg_class=20,
                 ratios=(0.5, 1, 2),
                 scales=(8, 16, 32)):
        extractor, classifier = decom_vgg16()



