import numpy as np
from vision.models.faster_rcnn.utils.bbox_tools import intersection_over_union, offset_scale


class AnchorTargetCreator:
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self,
                 bbox,
                 anchor,
                 img_size):
        n_anchor = len(anchor)

        # Filter anchors beyond image
        ix_inside = self._inside_index(anchor, *img_size)
        anchor = anchor[ix_inside]
        argmax_ious, label = self._create_label(ix_inside, anchor, bbox)

        loc = offset_scale(anchor, bbox[argmax_ious])

        label = self._unmap(label, n_anchor, ix_inside, fill=-1)
        loc = self._unmap(loc, n_anchor, ix_inside, fill=0)

        return loc, label

    def _create_label(self, ix_inside, anchor, bbox):
        # pos - 1; neg - 0; indeterminate - -1
        label = np.empty((len(ix_inside), ), dtype=np.int32)
        label.fill(-1)

        params = self._calc_ious(anchor, bbox, ix_inside)
        argmax_ious, max_ious, gt_argmax_ious = params

        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1

        # Subsample positive labels
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_ix = np.where(label == 1)[0]
        if len(pos_ix) > n_pos:
            disable_ix = np.random.choice(pos_ix,
                                          size=len(pos_ix)-n_pos,
                                          replace=False)
            label[disable_ix] = -1

        # Similarly for negative labels
        n_neg = self.n_sample - np.sum(label == 1)
        neg_ix = np.where(label == 0)[0]
        if len(neg_ix) > n_neg:
            disable_ix = np.random.choice(neg_ix,
                                          size=len(neg_ix)-n_neg,
                                          replace=False)
            label[disable_ix] = -1

        return argmax_ious, label

    @staticmethod
    def _calc_ious(anchor, bbox, ix_inside):
        # FIXME: remove _red
        # _red Redundant methods of original implementation
        from numpy.testing import assert_almost_equal

        ious = intersection_over_union(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious.max(axis=1)

        # FIXME: remove _red
        max_ious_red = ious[np.arange(len(ix_inside)), argmax_ious]
        assert_almost_equal(max_ious, max_ious_red)

        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious.max(axis=0)

        # FIXME: remove _red
        gt_max_ious_red = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        assert_almost_equal(gt_max_ious, gt_max_ious_red)

        # FIXME: remove _red
        gt_argmax_ious_red = np.where(ious == gt_max_ious)[0]
        assert_almost_equal(gt_argmax_ious, gt_argmax_ious_red)

        return argmax_ious, max_ious, gt_argmax_ious

    @staticmethod
    def _inside_index(anchor, height, width):
        return np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= height) &
            (anchor[:, 3] <= width)
        )[0]

    @staticmethod
    def _unmap(data: np.ndarray,
               count: int,
               index: np.ndarray,
               fill: int = 0) -> np.ndarray:
        if len(data.shape) == 1:
            ret = np.empty((count, ), dtype=data.dtype)
            ret.fill(fill)
            ret[index] = data
        else:
            ret = np.empty((count, )+data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[index, :] = data
        return ret
