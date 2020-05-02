import random
from vision.image_bbox import ImageBbox
import numpy as np


class RandomFlip:
    def __init__(self,
                 hor_rand: bool = True,
                 ver_rand: bool = False):
        """

        :param hor_rand: Flag to randomize horizontal flip
        :param ver_rand: Flag to randomize vertical flip
        """
        self.ver_rand = ver_rand
        self.hor_rand = hor_rand

    def __call__(self, img: ImageBbox) -> ImageBbox:
        v_flip, h_flip = False, False

        if self.ver_rand:
            v_flip = random.choice([True, False])
        if self.hor_rand:
            h_flip = random.choice([True, False])

        img.data = self.flip_img(img.data,
                                 v_flip,
                                 h_flip)
        img.bbox = self.flip_bbox(img.bbox,
                                  img.size,
                                  v_flip,
                                  h_flip)
        return img

    @staticmethod
    def flip_img(img: np.ndarray,
                 v_flip: bool,
                 h_flip: bool) -> np.ndarray:
        """

        :param img: (C, H, W)
        :param v_flip:
        :param h_flip:
        :return:
        """
        data = img.copy()

        if v_flip:
            data = data[:, ::-1, :]
        if h_flip:
            data = data[:, :, ::-1]

        return data

    @staticmethod
    def flip_bbox(bbox: np.ndarray,
                  size: tuple,
                  v_flip: bool = False,
                  h_flip: bool = False) -> np.ndarray:
        """

        :param bbox: (R, 4). Second axis: {y_min, x_min, y_max, x_max}
        :param size: (height, width) tuple
        :param v_flip:
        :param h_flip:
        :return:
        """
        height, width = size
        boxes = bbox.copy()

        if v_flip:
            boxes[:, [0, 2]] = height - boxes[:, [2, 0]]
        if h_flip:
            boxes[:, [1, 3]] = width - boxes[:, [3, 1]]

        return boxes




