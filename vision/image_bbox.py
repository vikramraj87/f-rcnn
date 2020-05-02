import numpy as np


class ImageBbox:
    @property
    def shape(self) -> tuple:
        """

        :return: (C, H, W)
        """
        return self.data.shape

    @property
    def size(self) -> tuple:
        """

        :return: (H, W)
        """
        return self.data.shape[1:]

    def __init__(self,
                 img_data: np.ndarray,
                 bbox: np.ndarray = None,
                 labels: np.ndarray = None,
                 difficult: bool = None):
        """

        :param img_data: (C, H, W)
        :param bbox: (R, 4). Second axis {y_min, x_min, y_max, x_max}
        :param labels:
        :param difficult: Is detection difficult
        """
        self.data = img_data
        self.bbox = bbox
        self.labels = labels
        self.difficult = difficult
