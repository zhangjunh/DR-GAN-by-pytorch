import os
import re
from operator import itemgetter
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

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
    >>> train_FD = FDDataset(root=root, train=True, single=False)
    >>> len(train_FD)
    500
    """
    def __init__(self, root, transform=None, train=True,
                    loader=default_loader, single=True):

        if train:
            path = os.path.join(root, 'train')
        else:
            path = os.path.join(root, 'test')

        imgs = make_dataset(path)

        if not single:
            imgs.sort(key=itemgetter('id'))
            imgs = split_with_same_id(imgs)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + path + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.train = train
        self.loader = loader
        self.single = single

    def dict2dict(self, sample):
        """
        input:(dict){'path':path, 'id': id, 'pose': pose, 'name': fname}
        output: (dict){'image': img, 'identity': identity, 'pose': pose, 'name': name}
        """
        identity = sample['id']
        pose = sample['pose']
        img = self.loader(sample['path'])
        name = '{0}_{1}'.format(sample['id'], sample['name'])

        if self.transform:
            img = self.transform(img)

        return {'image': img,
                'identity': identity,
                'pose': pose,
                'name': name}

    def __getitem__(self, idx):
        item = self.imgs[idx]
        if self.single:
            assert isinstance(item, dict)
            return self.dict2dict(item)
        else:
            assert isinstance(item, list)
            return [self.dict2dict(i) for i in item]

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    root = '/home/jaren/data/'
    transform = transforms.Compose([
        transforms.Scale((100, 100)),       #Switch to the transforms.Resize on the service
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_FD = FDDataset(root=root, train=True, transform=transform)
    test_FD = FDDataset(root=root, train=False, transform=transform)
    s = train_FD[78]
    show_sample(s)

    train_FD = FDDataset(root=root, train=True, transform=transform, single=False)
    samples = train_FD[100]
    for sample in samples:
        show_sample(sample)
