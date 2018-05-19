import os
import re
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
sys.path.append('/home/zhangjunhao/data')
from data_processing import *

class FDDataset(data.Dataset):
    """The class of  dataset.

    >>> root = '/home/jaren/data/'
    >>> train_FD = FDDataset(root=root, train=True)
    >>> test_FD = FDDataset(root=root, train=False)
    >>> len(train_FD)
    3566
    >>> len(test_FD)
    2027
    """
    def __init__(self, root, transform=None, train=True,
                    loader=default_loader):

        if train:
            path = os.path.join(root, 'train')
        else:
            path = os.path.join(root, 'test')

        imgs = make_dataset(path)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + path + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.train = train
        self.loader = loader

    def __getitem__(self, idx):
        idinfo = []
        for i in range(len(self.imgs)):
            idpath, idname = self.imgs[i]
            if get_id(idpath) == idx:
                idinfo.append((idpath, idname))
        self.imgs = idinfo
        lst = []
        for n in range(len(self.imgs)):
            path, fname = self.imgs[n]
            identity = get_id(path)
            pose = get_pose(path)
            img = self.loader(path)
            name = '{0}_{1}'.format(identity, fname)

            if self.transform:
                img = self.transform(img)
            lst.append({'image': img,
                        'identity': identity,
                        'pose': pose,
                        'name': name})
        return lst

    def __len__(self):
        return 450
