import torch
from torch import nn


class ROIPooling:
    def __init__(self, roi_size, spatial_scale):
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        self._adaptive_pool = nn.AdaptiveMaxPool2d((roi_size, roi_size),
                                                   return_indices=False)

    def __call__(self,
                 feature_map: torch.Tensor,
                 roi: torch.Tensor) -> torch.Tensor:
        """

        :param feature_map: (N, C, H, W)
        :param roi: (R, 5). Second axis: {i_x, y_min, x_min, y_max, x_max}
        :return: (R, C, roi_size, roi_size)
        """
        output = []
        scale = self.spatial_scale

        for r in roi:
            # Not useful now. Only batch size is 1
            img_ix = r[0].long()

            roi_l = (r[1:] * scale).long()
            y1, x1, y2, x2 = roi_l
            region = feature_map[img_ix, :, y1:(y2+1), x1:(x2+1)]
            region_p = self._adaptive_pool(region).unsqueeze(0)
            output.append(region_p)

        return torch.cat(output)
