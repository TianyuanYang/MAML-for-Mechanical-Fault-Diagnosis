import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .datasets import register


class MetaDataset(Dataset):
    def __init__(self, root_path, condition, split='train', n_batch=200, n_episode=4, n_way=5, k_shot=1, k_query=15,
                 length=1024):
        super(MetaDataset, self).__init__()

        split_dict = {
            'train': 'train',
            'val': 'val',
            'test': 'test',
        }
        split_tag = split_dict[split]

        self.resize = length
        # [10, 100, 1024, 1]
        self.data = np.load(os.path.join(root_path, condition + '_' + split_tag + '.npy'))
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        assert (k_shot + k_query) <= 20
        self.n_cls = self.data.shape[0]  # 10

        self.normalization()

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.max = np.max(self.data)
        self.min = np.min(self.data)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return self.n_batch * self.n_episode

    def __getitem__(self, index):
        shot, query = [], []
        cats = np.random.choice(self.n_cls, self.n_way, replace=False)
        for c in cats:
            c_shot, c_query = [], []
            data = self.data[c, :, :, :]  # [para_num, length, 1]
            idx_list = np.random.choice(data.shape[0], self.k_shot + self.k_query, replace=False)
            shot_idx, query_idx = idx_list[:self.k_shot], idx_list[-self.k_query:]
            for idx in shot_idx:
                c_shot.append(torch.from_numpy(data[idx, :, :]))
            for idx in query_idx:
                c_query.append(torch.from_numpy(data[idx, :, :]))
            shot.append(torch.stack(c_shot))
            query.append(torch.stack(c_query))

        shot = torch.cat(shot, dim=0)  # [n_way * k_shot, C, L]
        shot = torch.transpose(shot, 1, 2)
        query = torch.cat(query, dim=0)
        query = torch.transpose(query, 1, 2)
        cls = torch.arange(self.n_way)[:, None]
        shot_labels = cls.repeat(1, self.k_shot).flatten()
        query_labels = cls.repeat(1, self.k_query).flatten()
        return shot, query, shot_labels, query_labels


@register('cwru-dataset')
def cwru_dataset(root_path, condition, split='train', n_batch=200, n_episode=4, n_way=5, k_shot=1, k_query=15,
                 length=1024):
    return MetaDataset(root_path, condition, split, n_batch, n_episode, n_way, k_shot, k_query, length)


@register('lingang-dataset')
def lingang_dataset(root_path, condition, split='train', n_batch=200, n_episode=4, n_way=5, k_shot=1, k_query=15,
                 length=1024):
    return MetaDataset(root_path, condition, split, n_batch, n_episode, n_way, k_shot, k_query, length)


@register('su-dataset')
def su_dataset(root_path, condition, split='train', n_batch=200, n_episode=4, n_way=5, k_shot=1, k_query=15,
                 length=1024):
    return MetaDataset(root_path, condition, split, n_batch, n_episode, n_way, k_shot, k_query, length)