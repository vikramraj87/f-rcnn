from torch import nn, Tensor
from .feature_extractor import decom_vgg16
from .region_proposal_network import RegionProposalNetwork
from .roi_head import ROIHead


class Model(nn.Module):
    # Down sample 16x from input to conv5
    feat_stride = 16

    def __init__(self,
                 n_fg_class=20,
                 ratios=(0.5, 1, 2),
                 scales=(8, 16, 32)):
        super().__init__()
        extractor, classifier = decom_vgg16()

        self.n_class = n_fg_class + 1

        self.extractor = extractor

        self.rpn = RegionProposalNetwork(512,
                                         512,
                                         ratios=ratios,
                                         scales=scales,
                                         feat_stride=self.feat_stride)

        self.roi_head = ROIHead(n_class=self.n_class,
                                roi_size=7,
                                spatial_scale=1./self.feat_stride,
                                classifier=classifier)

    def forward(self, x: Tensor, scale=1.):
        img_size = x.shape[2:]

        feats = self.extractor(x)
        _, _, rois, roi_ix = self.rpn(feats, img_size, scale)
        roi_locs, roi_scores = self.roi_head(feats, rois, roi_ix)
        return roi_locs, roi_scores, rois, roi_ix





