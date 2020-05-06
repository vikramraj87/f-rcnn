import numpy as np
from ..image_bbox import ImageBbox
from skimage.transform import resize


class Resize:
    def __init__(self,
                 min_size=600,
                 max_size=1000):
        self.min = min_size
        self.max = max_size

    def __call__(self, img: ImageBbox) -> ImageBbox:
        channel, height, width = img.shape
        scale1 = self.min / min(height, width)
        scale2 = self.max / max(height, width)
        scale = min(scale1, scale2)

        new_shape = (channel, height * scale, width * scale)
        data = img.data.copy()
        data = resize(data,
                      new_shape,
                      mode='reflect',
                      anti_aliasing=False)

        if img.bbox is None:
            img.data = data
            return img

        boxes = self.resize_bbox(img.bbox,
                                 (height, width),
                                 (data.shape[1:]))

        img.data = data
        img.bbox = boxes

        return img

    @staticmethod
    def resize_bbox(bbox: np.ndarray,
                    orig_size: tuple,
                    new_size: tuple) -> np.ndarray:
        """

        :param bbox: (R, 4). Second axis: {y_min, x_min, y_max, x_max}
        :param orig_size: (height, width) of original image
        :param new_size: (height, width) of resized image
        :return: (R, 4)
        """
        boxes = bbox.copy()

        orig_size = np.array(orig_size, dtype=np.float32)
        new_size = np.array(new_size, dtype=np.float32)
        scale = new_size / orig_size
        scale = np.concatenate((scale, scale))

        return boxes * scale
