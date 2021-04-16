import time
import random
import torch
import cv2
import numpy as np
import os
import pickle
from collections import defaultdict
from tabulate import tabulate


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class MetricStorer:
    def __init__(self, metrics, raw_names, folds, path, esr_metric='test_loss', esr=3, esr_minimize=True):
        self.metrics = metrics
        self.folds = folds
        self.raw = {key: None for key in raw_names}
        self.best_raw = self.raw.copy()
        self.stored_metrics = defaultdict(list)
        self.temp_metrics = defaultdict(list)
        self.path = path
        self.time_counter = time.time()
        self.epoch_n = 1
        self.esr_metric = esr_metric
        self.esr = esr
        self.esr_minimize = esr_minimize

    def dump(self):
        with open(self.path, 'wb') as f:
            pickle.dump({
                'best_raw': self.best_raw,
                'stored_metrics': self.stored_metrics,
                'temp_metrics': self.temp_metrics
            }, f)

    def to_numpy(self, value):
        if isinstance(value, np.ndarray):
            return value
        else:
            if 'device' in value.__dir__():
                if value.device.type != 'cpu':
                    value = value.cpu()
            if 'requires_grad' in value.__dir__():
                if value.requires_grad:
                    value = value.detach()
            return value.numpy()

    def add_raw(self, name, batch, axis=0):
        if self.raw[name] is None:
            self.raw[name] = self.to_numpy(batch)
        else:
            self.raw[name] = np.concatenate([self.raw[name], batch], axis=axis)

    def clear_raw(self, name):
        self.raw[name] = None

    def move_best_raw(self):
        for key in self.raw.keys():
            self.best_raw[key] = self.raw[key]
            self.clear_raw(key)

    def apply_metric(self, name, *args, **kwargs):
        if isinstance(name, str):
            name = [name]
        for name_part in name:
            getattr(self, name_part)(*args, **kwargs)

    def new_epoch(self):
        for metric_name in self.temp_metrics.keys():
            self.stored_metrics[metric_name].append(np.mean(self.temp_metrics[metric_name]))
        self.temp_metrics = defaultdict(list)
        self.stored_metrics['epoch_n'].append(self.epoch_n)
        self.epoch_n += 1
        cur_time = time.time()
        self.stored_metrics['time'].append(int(cur_time - self.time_counter))
        self.time_counter = cur_time

    def add_loss(self, loss_value, foldname):
        if not isinstance(loss_value, float):
            loss_value = loss_value.detach().item()
        self.temp_metrics[foldname + '_loss'].append(loss_value)

    def print_last(self, inplace=True, esr=True):
        vals = [
            self.stored_metrics['epoch_n'][-1],
            self.stored_metrics['time'][-1],
            self.stored_metrics['test_loss'][-1],
            self.stored_metrics['train_loss'][-1],
            self.stored_metrics['test_acc'][-1],
            self.stored_metrics['train_acc'][-1]
        ]
        headers = ['epoch_n', 'time', 'test_loss', 'train_loss', 'test_acc', 'train_acc']
        if inplace:
            print(tabulate([vals], headers))
        target_vals = self.stored_metrics[self.esr_metric]
        is_best = min(target_vals) == target_vals[-1] if self.esr_minimize else max(target_vals) == target_vals[-1]
        if esr:
            target_index = target_vals.index(min(target_vals) if self.esr_minimize else max(target_vals) + 1)
            non_incr_indexes = len(target_vals) - target_index
            print('Current non increasing: {}, current esr: {}'.format(non_incr_indexes, self.esr))
            if non_incr_indexes > self.esr:
                to_break = True
            else:
                to_break = False
        else:
            to_break = False
        return '\n' + tabulate([vals], headers), to_break, is_best

    def acc(self, pred, real, foldname):
        assert len(pred) == len(real)
        pred, real = self.to_numpy(pred), self.to_numpy(real)
        res = np.mean(pred == real)
        self.temp_metrics[foldname + '_acc'].append(res)