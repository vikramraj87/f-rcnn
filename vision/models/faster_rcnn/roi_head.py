import torch
from torch import nn


class ROIHead(nn.Module):
    def __init__(self,
                 n_class: int,
                 roi_size: int,
                 spatial_scale,
                 classifier):
        super().__init__()

        self.classifier = classifier
        self.loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self._weights_init()

    def forward(self,
                pooled_regions: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param pooled_regions: (R, C, roi_size, roi_size)
        :return: (n_class * 4), (n_class)
        """
        x = self.classifier(pooled_regions)
        locs = self.loc(x)
        scores = self.score(x)
        return locs, scores

    def _weights_init(self):
        self.loc.weight.data.normal_(0, 0.001)
        self.loc.bias.data.zero_()

        self.score.weight.data.normal_(0, 0.01)
        self.score.bias.data.zero_()

