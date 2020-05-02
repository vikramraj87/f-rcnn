import unittest
import numpy as np
from numpy import testing
from ..anchor_gen import AnchorGenerator


class AnchorGeneratorTests(unittest.TestCase):
    def test_anchors(self):
        base = self.generate_anchor_base()
        exp = self._enumerate_shifted_anchor(base, 16, 50, 60)

        gen = AnchorGenerator()
        result = gen(50, 60)
        testing.assert_almost_equal(exp, result)
        result2 = gen(50, 60)
        testing.assert_almost_equal(exp, result2)

    def test_num_anchors(self):
        gen = AnchorGenerator(ratios=(1, 2, 3, 4),
                              scales=(8, 16, 24, 32))
        self.assertEqual(16, gen.n_anchors)

    @staticmethod
    def _corners(boxes: np.ndarray) -> np.ndarray:
        top_right = boxes[:, :2] - boxes[:, 2:] * 0.5
        bot_left = boxes[:, :2] + boxes[:, 2:] * 0.5
        return np.concatenate((top_right, bot_left), axis=1)

    @staticmethod
    def generate_anchor_base(base_size=16,
                             ratios=(0.5, 1, 2),
                             anchor_scales=(8, 16, 32)):
        center_pt = np.array((base_size / 2., base_size / 2.),
                             dtype=np.float32)
        center = np.zeros((len(ratios) * len(anchor_scales), 1),
                          dtype=center_pt.dtype)

        scales = np.array(anchor_scales, dtype=center_pt.dtype) * base_size

        r = np.array(ratios, dtype=center_pt.dtype)[:, np.newaxis]

        dims = scales[np.newaxis] * (np.sqrt(r), np.sqrt(1. / r))  # (2, 3, 3)
        dims = dims.reshape(2, -1).transpose(1, 0)

        anchors_cd = np.concatenate((center + center_pt, dims), axis=1)

        return AnchorGeneratorTests._corners(anchors_cd)

    @staticmethod
    def _enumerate_shifted_anchor(anchor_base,
                                  feat_stride,
                                  height,
                                  width):
        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x = np.arange(0, width * feat_stride, feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shift = np.stack((shift_y.ravel(),
                          shift_x.ravel(),
                          shift_y.ravel(),
                          shift_x.ravel()),
                         axis=1)

        anchor = anchor_base.reshape((1, -1, 4)) + shift.reshape((-1, 1, 4))
        anchor = anchor.reshape((-1, 4)).astype(np.float32)
        return np.ascontiguousarray(anchor)


if __name__ == '__main__':
    unittest.main()
