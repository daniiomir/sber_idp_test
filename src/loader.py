import os
import random
import numpy as np
import albumentations as albu
from torch.utils.data.dataset import Dataset
from src.tools import read_image
from config import CNFG


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


test_transforms = albu.Compose([
    albu.Resize(CNFG['img_size'], CNFG['img_size']),
    albu.Normalize()
])


class WoofDataset(Dataset):
    def __init__(self, files, class_index, data_path, transforms, mode):
        self.files = files
        self.class_index = class_index
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = read_image(os.path.join(self.data_path, self.files[index]))
        augm_img = self.transforms(image=img)['image']
        if self.mode == 'train':
            return augm_img, self.class_index
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
