import os
import os.path as osp
import cv2
import numpy as np
import torch
from PIL import Image

from .base import _BaseClsDataset, _AdvClsDataset


class VOCWSSS(_AdvClsDataset):
    """
    PASCAL VOC Segmentation dataset
    """

    def __init__(self, year=2012, **kwargs):
        self.year = year
        super(VOCWSSS, self).__init__(**kwargs)

    def _set_files(self):
        self.root = osp.join(self.root, "VOC{}".format(self.year))
        self.image_dir = osp.join(self.root, "JPEGImages")
        self.label_dir = osp.join(self.data_dir, "datasets", "voc")
        self.mask_dir = osp.join(self.label_dir, "masks")

        if self.split in ['train', 'train_aug']:
            file_list = osp.join(
                self.label_dir, "list", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip().split(' ') for id_ in file_list]
            self.files, _ = list(zip(*file_list))

            self.labels = np.load(osp.join(self.label_dir, "cls_labels.npy"), allow_pickle=True).item()
            if self.fg_path is None:
                self.fgs = np.load(osp.join(self.label_dir, "fg_init.npy"), allow_pickle=True).item()
            else:
                self.fgs = np.load(self.fg_path, allow_pickle=True).item()

        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][1:])
        mask_path = osp.join(self.mask_dir, image_id + '.png')
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = self.labels[image_id]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        if self.fgs is None:
            fg = None
        else:
            fg = self.fgs[image_id]
        return image_id, image, label, mask, fg
