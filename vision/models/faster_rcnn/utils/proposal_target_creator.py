import numpy as np
from .bbox_tools import intersection_over_union, offset_scale


class ProposalTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self,
                 roi: np.ndarray,
                 bbox: np.ndarray,
                 label: np.ndarray,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox = bbox.shape[0]

        iou = intersection_over_union(roi, bbox)
        gt_assignment = np.argmax(iou, axis=1)
        max_iou = np.max(iou, axis=1)

        # Range(0, n_labels) -> Range(1, n_labels+1). 0 -> bg
        gt_roi_label = label[gt_assignment] + 1

        n_pos_max = np.round(self.n_sample * self.pos_ratio)
        pos_ix = np.where(max_iou >= self.pos_iou_thresh)[0]
        n_pos = int(min(n_pos_max, len(pos_ix)))
        if len(pos_ix) > n_pos:
            pos_ix = np.random.choice(pos_ix, n_pos, replace=False)

        neg_ix = np.where((max_iou < self.neg_iou_thresh_hi) &
                          (max_iou >= self.neg_iou_thresh_lo))[0]
        n_neg = self.n_sample - n_pos
        if len(neg_ix) > n_neg:
            neg_ix = np.random.choice(neg_ix, n_neg, replace=False)

        keep_index = np.append(pos_ix, neg_ix)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[n_pos:] = 0
        sample_roi = roi[keep_index]

        gt_roi_loc = offset_scale(sample_roi, bbox[gt_assignment[keep_index]])

        # Normalize gt_roi_loc
        # FIXME: Significance of Normalization
        # To follow original implementation, uncomment
        # gt_roi_loc -= np.array(loc_normalize_mean, np.float32)
        # gt_roi_loc /= np.array(loc_normalize_std, np.float32)

        return sample_roi, gt_roi_loc, gt_roi_label






