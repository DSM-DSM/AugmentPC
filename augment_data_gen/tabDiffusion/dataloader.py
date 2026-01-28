import torch
from torch.utils.data import Dataset
import math


class FastTensorDataLoader:
    """优化版的Tensor数据加载器"""
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = math.ceil(self.dataset_len / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class TabularDataset(Dataset):
    """处理表格数据的Dataset类"""
    def __init__(self, X_num, X_cat, y):
        """
        Args:
            X_num: 数值特征 (numpy array)
            X_cat: 类别特征 (numpy array)
            y: 标签 (numpy array)
        """
        self.X_num = torch.FloatTensor(X_num) if X_num is not None else None
        self.X_cat = torch.LongTensor(X_cat) if X_cat is not None else None
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_num = self.X_num[idx] if self.X_num is not None else torch.tensor([])
        x_cat = self.X_cat[idx] if self.X_cat is not None else torch.tensor([])
        y = self.y[idx]
        return torch.cat([x_num, x_cat.float()]), y


# 创建子集
class SubsetWithAttributes(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

    @property
    def X_num(self):
        return torch.stack([self.subset[i][0][:5] for i in range(len(self))])

    @property
    def X_cat(self):
        return torch.stack([self.subset[i][0][5:] for i in range(len(self))])

    @property
    def y(self):
        return torch.stack([self.subset[i][1] for i in range(len(self))])
