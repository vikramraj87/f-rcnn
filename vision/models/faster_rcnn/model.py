from torch import nn, Tensor, no_grad
from .feature_extractor import decom_vgg16
from .region_proposal_network import RegionProposalNetwork
from .roi_head import ROIHead
import numpy as np
from .utils.bbox_tools import non_max_supression
from .utils.device import device
from .utils.array_tools import to_tensor, to_numpy
from ...image_bbox import ImageBbox
from ...transforms.resize import Resize
from .utils.bbox_tools import from_offset_scale
import torch



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

        self.nms_thresh = None
        self.score_thresh = None

        self.use_preset("evaluate")

    def use_preset(self, preset):
        if preset == "visualize":
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == "evaluate":
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError("Invalid preset.")

    def forward(self, x: Tensor, scale=1.):
        img_size = x.shape[2:]

        feats = self.extractor(x)
        _, _, rois, roi_ix = self.rpn(feats, img_size, scale)
        roi_locs, roi_scores = self.roi_head(feats, rois, roi_ix)
        return roi_locs, roi_scores, rois, roi_ix

    def _supress(self, raw_cls_bbox: np.ndarray, raw_prob: np.ndarray):
        bbox = list()
        label = list()
        score = list()

        # Skip background
        for cls in range(1, self.n_class):
            bbox_cls = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, cls, :]
            prob_cls = raw_prob[:, cls]

            mask = prob_cls > self.score_thresh
            bbox_cls = bbox_cls[mask]
            prob_cls = prob_cls[mask]

            if len(prob_cls.shape) == 1:
                print("prob_cls is a 1-dim array.")
                print(f"Remove ravel() in model._supress() @ model.py")
            # Order boxes based on probability as nms expects
            # boxes expect them in order of descending probability
            order = prob_cls.ravel().argsort()[::-1].astype(np.int32)
            bbox_cls = bbox_cls[order]
            keep = non_max_supression(bbox_cls, self.nms_thresh)

            # The labels are in [0, self.n_class - 2].
            bbox.append(bbox_cls[keep])
            label.append((cls - 1) * np.ones((len(keep),), dtype=np.int32))
            score.append(prob_cls[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score

    @no_grad()
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset("visualize")
            prepared_imgs = list()
            sizes = list()
            resize_tf = Resize()
            for img in imgs:
                size = img.shape[1:]
                data = to_numpy(img)
                resized = resize_tf(ImageBbox(data)).data
                prepared_imgs.append(resized)

                # FIXME: Check original implementation
                # size should represent size after resize or original
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()

        for img, size in zip(prepared_imgs, sizes):
            img = to_tensor(img[None]).float()
            scale = img.shape[3] / size[1]

            # Assumption: batch_size is 1
            roi_locs, roi_scores, rois, _ = self(img, scale=scale)
            roi_scores = roi_scores.detach().cpu()
            roi_locs = roi_locs.detach().cpu()

            rois = to_tensor(rois) / scale


            # FIXME: Didn't normalize as in original implementation

            rois = rois.view(-1, 1, 4).expand_as(roi_locs)
            cls_bbox = from_offset_scale(to_numpy(rois).reshape((-1, 4)),
                                         to_numpy(roi_locs).reshape(-1, 4))
            cls_bbox = to_tensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class, 4)
            cls_bbox[:, ::2] = torch.clamp(cls_bbox[:, ::2], 0, size[0])
            cls_bbox[:, 1::2] = torch.clamp(cls_bbox[:, 1::2], 0, size[1])

            prob = nn.functional.softmax(to_tensor(roi_scores), dim=1)
            prob = to_numpy(prob)

            raw_cls_bbox = to_numpy(cls_bbox)
            raw_prob = to_numpy(prob)

            bbox, label, score = self._supress(raw_cls_bbox, raw_prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset("evaluate")
        self.train()
        return bboxes, labels, scores





