from torch.utils.data import Dataset
from pathlib import Path
import xml.etree.ElementTree as ET


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
                 use_difficult=False):

        id_list_file = data_dir / f"ImageSets/Main/{split}.txt"

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.annot_dir = data_dir / "Annotations"
        self.image_dir = data_dir / "JPEGImages"

        self.use_difficult = use_difficult

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id_ = self.ids[item]

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
            bbox = []




