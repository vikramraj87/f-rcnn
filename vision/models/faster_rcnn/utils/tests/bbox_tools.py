import unittest
import numpy as np
from ..bbox_tools import intersection_over_union as iou, \
    from_offset_scale, non_max_supression, offset_scale
from numpy.testing import assert_almost_equal


class BBoxToolsTest(unittest.TestCase):
    def test_iou(self):
        boxes_a = np.array([(2, 4, 100, 16),
                            (18, 8, 38, 20),
                            (78, 100, 140, 150),
                            (200, 30, 270, 200)], dtype=np.float32)

        boxes_b = np.array([(12, 8, 120, 38),
                            (300, 100, 370, 280),
                            (12, 4, 500, 600),
                            (450, 200, 600, 450),
                            (100, 100, 300, 350)], dtype=np.float32)

        expected = np.array([[0.18965517, 0., 0.00362927, 0., 0.],
                             [0.07407407, 0., 0.00082517, 0., 0.],
                             [0., 0., 0.01065849, 0., 0.03913894],
                             [0., 0., 0.04091484, 0., 0.12750456]],
                            dtype=np.float32)

        result = iou(boxes_a, boxes_b)
        assert_almost_equal(result, expected)

    def test_offset_scale(self):
        boxes_a = np.array([(2, 4, 100, 16),
                            (18, 8, 38, 20),
                            (78, 100, 140, 150),
                            (200, 30, 270, 200)], dtype=np.float32)

        boxes_b = np.array([(12, 8, 120, 38),
                            (300, 100, 370, 280),
                            (12, 4, 500, 600),
                            (450, 200, 600, 450),
                            (100, 100, 300, 350)], dtype=np.float32)

        expected = np.array([[0.15306122, 1.08333333, 0.09716375, 0.91629073],
                             [15.35, 14.66666667, 1.25276297, 2.7080502],
                             [2.37096774, 3.54, 2.06318102, 2.47821766],
                             [4.14285714, 1.23529412, 0.76214005, 0.38566248]],
                            dtype=np.float32)

        result = offset_scale(boxes_a, boxes_b[:-1])
        assert_almost_equal(result, expected)

    def test_from_offset(self):
        exp = np.array([(12.000002533197403, 8.000000476837158, 119.99999776482582, 38.00000047683716),
                        (300.0000100135803, 99.99999809265137, 370.00000524520874, 280.00000953674316),
                        (12.00001859664917, 4.0000200271606445, 499.9999670982361, 599.9999761581421),
                        (449.9999976158142, 200.0, 599.9999928474426, 449.9999952316284)], dtype=np.float32)

        boxes_a = np.array([(2, 4, 100, 16),
                            (18, 8, 38, 20),
                            (78, 100, 140, 150),
                            (200, 30, 270, 200)], dtype=np.float32)

        offset = np.array([[0.15306122, 1.08333333, 0.09716375, 0.91629073],
                           [15.35, 14.66666667, 1.25276297, 2.7080502],
                           [2.37096774, 3.54, 2.06318102, 2.47821766],
                           [4.14285714, 1.23529412, 0.76214005, 0.38566248]],
                          dtype=np.float32)

        result = from_offset_scale(boxes_a, offset)
        assert_almost_equal(result, exp, decimal=4)

    def test_nms(self):
        boxes_b = [(12, 8, 120, 38),
                   (300, 100, 370, 280),
                   (14, 7, 100, 50),
                   (450, 200, 600, 450),
                   (100, 100, 300, 350)]
        boxes_b = np.array(boxes_b)
        ix = non_max_supression(boxes_b, 0.5)
        assert_almost_equal(ix, np.array([0, 1, 3, 4], dtype=np.int32))


if __name__ == '__main__':
    unittest.main()
