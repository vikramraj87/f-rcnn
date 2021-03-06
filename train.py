from vision.datasets.voc import VOC
from pathlib import Path
from torchvision.transforms import Compose
from vision.transforms.resize import Resize
from vision.transforms.random_flip import RandomFlip
from vision.transforms.normalized_tensor import NormalizedTensor
from torch.utils.data import DataLoader


def train():
    data_dir = Path("./data/VOCdevkit/VOC2007").resolve()
    transforms = [Resize(600, 1000),
                  RandomFlip(True, False),
                  NormalizedTensor()]

    train_dataset = VOC(data_dir,
                        transform=Compose(transforms))
    train_dl = DataLoader(train_dataset,
                          batch_size=1,
                          shuffle=True)

    # No random flip for tests dataset
    del transforms[1]
    test_dataset = VOC(data_dir,
                       split="test",
                       transform=Compose(transforms))
    test_dl = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=True)

    sample = train_dataset[2]
    image = sample["image"]
    bbox = sample["bbox"]
    label = sample["labels"]
    difficult = sample["difficult"]

    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Bbox shape: {bbox.shape}")
    print(f"Labels shape: {label.shape}")
    print(f"Difficult shape: {difficult.shape}")


if __name__ == '__main__':
    train()