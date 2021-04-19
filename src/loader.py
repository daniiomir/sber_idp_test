import os
import torch
import random
import numpy as np
import albumentations as albu
from torch.utils.data.dataset import Dataset
from src.tools import read_image
from src.config import CNFG
from sklearn.model_selection import train_test_split


train_transforms = albu.Compose([
    albu.Resize(CNFG['img_size'], CNFG['img_size']),
    albu.RandomBrightnessContrast(),
    albu.ColorJitter(),
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ShiftScaleRotate(),
    albu.Rotate(),
    albu.Normalize()
])


val_transforms = albu.Compose([
    albu.Resize(CNFG['img_size'], CNFG['img_size']),
    albu.Normalize()
])


def split_img_list(img_name_list, test_size):
    res = tuple(train_test_split(range(len(img_name_list)), test_size=test_size))
    files = np.array(img_name_list)
    return files[res[0]], files[res[1]]


class WoofDataset(Dataset):
    def __init__(self, files, labels, data_path, transforms, mode):
        self.files = files
        self.labels = labels
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = read_image(os.path.join(self.data_path, self.files[index]))
        augm_img = self.transforms(image=img)['image']
        if self.mode == 'train':
            return augm_img, torch.from_numpy(np.asarray(self.labels[index]))
        else:
            return augm_img


class MultipleDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = [*datasets]
        self.mapper = []
        for dataset_n, dataset in enumerate(datasets):
            for index in range(len(dataset)):
                self.mapper.append((dataset_n, index))
        random.shuffle(self.mapper)
        self.mapper = {key: item for key, item in enumerate(self.mapper)}

    def __len__(self):
        return np.sum([len(_) for _ in self.datasets])

    def __getitem__(self, item):
        dataset_n, index = self.mapper[item]
        return self.datasets[dataset_n][index]
