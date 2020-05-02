from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from vision.image_bbox import ImageBbox


class VOC(Dataset):
    label_names = (
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    )

    def __init__(self,
                 data_dir: Path,
                 split="trainval",
                 use_difficult=False,
                 transform=None):

        id_list_file = data_dir / f"ImageSets/Main/{split}.txt"

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.annot_dir = data_dir / "Annotations"
        self.image_dir = data_dir / "JPEGImages"

        self.use_difficult = use_difficult
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id_ = self.ids[item]
        bbox, label, difficult = self._annotations(id_)
        img = self._image(id_)

        img_bbox = ImageBbox(img, bbox, label, difficult)

        if self.transform:
            img_bbox = self.transform(img_bbox)

        return img_bbox

    def _annotations(self, id_: str):
        xml = self.annot_dir / f"{id_}.xml"

        bbox = list()
        label = list()
        difficult = list()

        annot = ET.parse(xml)
        for obj in annot.findall("object"):
            diff = int(obj.find("difficult").text)
            if not self.use_difficult and diff == 1:
                continue
            difficult.append(diff)

            bbox_xml = obj.find("bndbox")
            box = [int(bbox_xml.find(tag).text) - 1
                   for tag in ('ymin', 'xmin', 'ymax', 'xmax')]
            bbox.append(box)

            name = obj.find("name").text.lower().strip()
            lbl = VOC.label_names.index(name)
            label.append(lbl)

        bbox = np.stack(bbox).astype(np.float32)
        # TODO: uint8?
        label = np.array(label).astype(np.int32)
        # TODO: Check support for np.bool in pytorch
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        return bbox, label, difficult

    def _image(self, id_: str, dtype=np.float32) -> np.ndarray:
        file = self.image_dir / f"{id_}.jpg"
        img = None

        with Image.open(file) as f:
            img = f.convert("RGB")
            img = np.asarray(img, dtype=dtype)

        if img.ndim == 2:
            # (H, W) -> (1, H, W)
            return img[np.newaxis, :]

        # (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))




