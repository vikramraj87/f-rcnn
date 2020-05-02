import unittest
from ..random_flip import RandomFlip
import numpy as np
from numpy import testing


class ImageFlipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.img = np.arange(27).reshape(-1, 3, 3)
        self.bbox = np.arange(32).reshape(-1, 4)

    def test_img_h_flip(self):
        expected = self.img[:, :, ::-1]
        result = RandomFlip.flip_img(self.img, False, True)
        testing.assert_almost_equal(result, expected)

    def test_img_v_flip(self):
        expected = self.img[:, ::-1, :]
        result = RandomFlip.flip_img(self.img, True, False)
        testing.assert_almost_equal(result, expected)

    def test_img_both_flip(self):
        expected = self.img[:, ::-1, :]
        expected = expected[:, :, ::-1]
        result = RandomFlip.flip_img(self.img, True, True)
        testing.assert_almost_equal(result, expected)

    def test_bbox_v_flip(self):
        expected = np.array(([[38, 1, 40, 3],
                              [34, 5, 36, 7],
                              [30, 9, 32, 11],
                              [26, 13, 28, 15],
                              [22, 17, 24, 19],
                              [18, 21, 20, 23],
                              [14, 25, 16, 27],
                              [10, 29, 12, 31]]))
        result = RandomFlip.flip_bbox(self.bbox, (40, 40), True, False)
        testing.assert_almost_equal(result, expected)

    def test_bbox_h_flips(self):
        expected = np.array([[0, 37, 2, 39],
                             [4, 33, 6, 35],
                             [8, 29, 10, 31],
                             [12, 25, 14, 27],
                             [16, 21, 18, 23],
                             [20, 17, 22, 19],
                             [24, 13, 26, 15],
                             [28, 9, 30, 11]])
        result = RandomFlip.flip_bbox(self.bbox, (40, 40), False, True)
        testing.assert_almost_equal(result, expected)

    def test_bbox_both_flip(self):
        expected = np.array(([[38, 37, 40, 39],
                              [34, 33, 36, 35],
                              [30, 29, 32, 31],
                              [26, 25, 28, 27],
                              [22, 21, 24, 23],
                              [18, 17, 20, 19],
                              [14, 13, 16, 15],
                              [10, 9, 12, 11]]))
        result = RandomFlip.flip_bbox(self.bbox, (40, 40), True, True)
        testing.assert_almost_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
