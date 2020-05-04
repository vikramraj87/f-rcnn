import numpy as np
from .bbox_tools import from_offset_scale, non_max_supression


class ProposalCreator:
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.nms_thresh = nms_thresh

        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms

        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

        self.min_size = min_size

    def __call__(self,
                 training: bool,
                 loc: np.ndarray,
                 score: np.ndarray,
                 anchor: np.ndarray,
                 img_size: tuple,
                 scale=1.):

        if training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal by applying proposal offset
        roi = from_offset_scale(anchor, loc)

        # Clip proposals to image
        height, width = img_size
        roi[:, ::2] = np.clip(roi[:, ::2], 0, height)
        roi[:, 1::2] = np.clip(roi[:, 1::2], 0, width)

        # Remove proposals less than min_size
        min_size = self.min_size * scale
        roi_h = roi[:, 2] - roi[:, 0]
        roi_w = roi[:, 3] - roi[:, 1]
        keep = np.where((roi_h >= min_size) &
                        (roi_w >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort proposals to score
        # Take top n_pre_nms
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]

        keep = non_max_supression(roi, self.nms_thresh)

        if n_post_nms > 0:
            keep = keep[:n_post_nms]

        roi = roi[keep]
        return roi


