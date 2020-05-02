import numpy as np
from .bbox_tools import corners


class AnchorGenerator:
    @property
    def n_anchors(self):
        return len(self._ratios) * len(self._scales)

    def __init__(self,
                 feat_stride=16,
                 ratios=(0.5, 1, 2),
                 scales=(8, 16, 32)):
        self._scales = scales
        self._ratios = ratios
        self._feat_stride = feat_stride
        self._dims = self._dimensions()
        self._cache_storage = dict()

    def __call__(self, height, width):
        """ Generate anchors """
        cached_anchors = self._cache(height, width)
        if cached_anchors is not None:
            return cached_anchors

        points = self._points(height, width)

        # Create a mesh grid of indices for points and dimensions,
        # and ravel to create a permutation of points and
        # dim indices. Finally concatenate dims and points
        # based on the indices.
        ix_points = np.arange(len(points), dtype=np.uint16)
        ix_dims = np.arange(len(self._dims), dtype=np.uint16)
        dd, pp = np.meshgrid(ix_dims, ix_points)
        anchors_cd = np.concatenate((points[pp.ravel()],
                                     self._dims[dd.ravel()]),
                                    axis=1)
        anchors = corners(anchors_cd)
        self._set_cache(anchors, height, width)
        return anchors

    def _dimensions(self):
        """ Create all possible dimesions based on ratios and scales"""
        scales = np.array(self._scales, dtype=np.float32) * self._feat_stride
        ratios = np.array(self._ratios, dtype=scales.dtype)[:, np.newaxis]
        ratios_root = np.sqrt(ratios)

        dims = scales[np.newaxis] * (ratios_root, 1. / ratios_root)
        return dims.reshape(2, -1) \
                   .transpose(1, 0)

    def _points(self, height, width):
        """ Create all possible center points based on height and width """
        y = np.arange(height, dtype=np.float32) * self._feat_stride
        y += self._feat_stride * 0.5

        x = np.arange(width, dtype=y.dtype) * self._feat_stride
        x += self._feat_stride * 0.5

        # Create a mesh grid and ravel to create a permutation of y and
        # x values. Finally stack them.
        xx, yy = np.meshgrid(x, y)
        return np.stack((yy.ravel(), xx.ravel())) \
                 .transpose()

    def _cache(self, height, width):
        key = f"{height}_{width}"
        return self._cache_storage.get(key, None)

    def _set_cache(self, anchors, height, width):
        key = f"{height}_{width}"
        self._cache_storage[key] = anchors
