import torch
from vision.image_bbox import ImageBbox
from torchvision.transforms import Normalize


class NormalizedTensor:
    def __init__(self,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: ImageBbox):
        """ Convert ImageBbox to Dict of Tensors """

        # Normalize the data
        data = torch.from_numpy(image.data)
        data /= 255.
        data = Normalize(mean=self.mean, std=self.std)

        return {
            "image": data,
            "bbox": torch.from_numpy(image.bbox),
            "labels": torch.from_numpy(image.labels),
            "difficult": torch.from_numpy(image.difficult)
        }
