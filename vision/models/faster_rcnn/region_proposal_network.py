from torch import nn, Tensor
from .utils.anchor_gen import AnchorGenerator
from .utils.proposal_creator import ProposalCreator
import numpy as np


class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels=512,
                 mid_channels=512,
                 ratios=(0.5, 1, 2),
                 scales=(8, 16, 32),
                 feat_stride=16):
        super().__init__()

        self.anchor_gen = AnchorGenerator(feat_stride=feat_stride,
                                          ratios=ratios,
                                          scales=scales)
        self.feat_stride = feat_stride
        self.proposal_creator = ProposalCreator()

        self.n_anchor = self.anchor_gen.n_anchors

        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.score = nn.Conv2d(mid_channels,
                               self.n_anchor * 2,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.loc = nn.Conv2d(mid_channels,
                             self.n_anchor * 4,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        self._normal_init(self.conv1, 0, 0.01)
        self._normal_init(self.score, 0, 0.01)
        self._normal_init(self.loc, 0, 0.01)

    def forward(self, x: Tensor, img_size, scale=1.):
        batch_size, _, height, width = x.shape
        anchors = self.anchor_gen(height, width)

        mid = self.conv1(x)
        mid = nn.functional.relu(x)

        locs = self.loc(mid)  # (n_batch, n_anchor * 4, h, w)
        locs = locs.permute((0, 2, 3, 1)) \
            .contiguous() \
            .view(batch_size, -1, 4)  # (n_batch, h * w * n_anchor, 4)

        scores = self.score(mid)
        scores = scores.permute((0, 2, 3, 1)).contiguous()

        softmax_scores = nn.functional.softmax(
            scores.view(batch_size, height, width, self.n_anchor, 2),
            dim=4
        )

        fg_scores = softmax_scores[..., 1].contiguous()
        fg_scores = fg_scores.view(batch_size, -1)
        scores = scores.view(batch_size, -1, 2)

        rois = list()
        roi_ix = list()

        for i in range(batch_size):
            roi = self.proposal_creator(
                self.training,
                locs[i].detach().cpu().numpy(),
                scores[i].detach().cpu().numpy(),
                anchors,
                img_size,
                scale=scale
            )
            batch_ix = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_ix.append(batch_ix)

        rois = np.concatenate(rois, axis=0)
        roi_ix = np.concatenate(roi_ix, axis=0)

        return locs, scores, rois, roi_ix, anchors

    @staticmethod
    def _normal_init(layer, mean, std):
        layer.weight.data.normal_(mean, std)
        layer.bias.data.zero_()
