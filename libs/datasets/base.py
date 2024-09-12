import numpy as np
import torch
from torch.utils import data

import libs.datasets.transforms as transforms


class _BaseClsDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(
            self,
            root,  # dataset path
            data_dir,
            split,
            mean_rgb,
            std_rgb,
            target_size,
            augmentation=True

    ):
        self.root = root
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        self.augmentation = augmentation
        self.files = []
        self._set_files()

        self.RandRescaleCrop = transforms.RandRescaleCrop(target_size=target_size)
        self.RandAugment = transforms.RandAugment()
        self.TransformToTensor = transforms.TransformToTensor(
            target_size=(target_size, target_size),
            mean=mean_rgb,
            std=std_rgb
        )

    def _set_files(self):
        raise NotImplementedError()

    def _load_data(self, image_id):
        raise NotImplementedError()

    def augment(self, image, label=None, fg=None):
        image, label, fg = self.RandRescaleCrop(image, label, fg)
        image, label = self.RandAugment(image, label)
        return image, label, fg

    def transform(self, image, label=None):
        image, label = self.TransformToTensor(image, label)
        return image, label

    def __getitem__(self, index):
        image_id, image, cls_label = self._load_data(index)
        if self.augmentation:
            image, _, _ = self.augment(image)

        image, _ = self.transform(image)

        return image_id, image, cls_label

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class _AdvClsDataset(_BaseClsDataset):
    """
    Base dataset class
    """

    def __init__(
            self,
            fg_path=None,
            **kwargs
    ):
        self.fg_path = fg_path
        super(_AdvClsDataset, self).__init__(**kwargs)

    def _set_files(self):
        raise NotImplementedError()

    def _load_data(self, image_id):
        raise NotImplementedError()

    def __getitem__(self, index):
        image_id, image, cls_label, mask, fg = self._load_data(index)

        if self.augmentation:
            image, mask, fg = self.augment(image, mask, fg)
        image, mask = self.transform(image, mask)
        fg = torch.tensor(fg)

        return image_id, image, cls_label, mask, fg

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class _BaseSegDataset(data.Dataset):
    """
    Base dataset class
    """

    def __init__(
            self,
            root,  # dataset path
            split,
            mean_rgb,
            std_rgb,
            target_size,
            augmentation=True

    ):
        self.root = root
        self.split = split
        self.target_size = target_size
        self.augmentation = augmentation
        self.files = []
        self._set_files()

    def _set_files(self):
        raise NotImplementedError()

    def _load_data(self, image_id):
        raise NotImplementedError()

    @staticmethod
    def augment(image, label=None, fg=None):
        image, label, fg = transforms.RandRescaleCrop(image, label, fg)
        image, label = transforms.RandAugment(image, label)
        return image, label, fg

    @staticmethod
    def transform(image, label=None):
        image, label = transforms.TransformToTensor(image, label)
        return image, label

    def __getitem__(self, index):
        image_id, image, cls_label = self._load_data(index)
        if self.augmentation:
            image, _, _ = self.augment(image)

        image, _ = self.transform(image)

        return image_id, image, cls_label

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
