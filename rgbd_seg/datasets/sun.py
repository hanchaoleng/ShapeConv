import logging
import os

import cv2
import numpy as np

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class SUNDataset(BaseDataset):
    def __init__(self, root, imglist_name, classes=37, channels=None, transform=None,
                 multi_label=False):
        if multi_label:
            raise ValueError('multi label training is only '
                             'supported by using COCO data form')
        super().__init__()
        self.transform = transform
        if channels is None:
            channels = ['rgb', 'hha']
        imglist_fp = os.path.join(root, imglist_name)
        self.imglist = self.read_imglist(imglist_fp)

        logger.debug('Total of images is {}'.format(len(self.imglist)))

        self.root = root
        self.channels = channels

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        inputs = []
        if 'rgb' in self.channels:
            img_fp = os.path.join(self.root, 'data', imgname, 'image.jpg')
            img = cv2.imread(img_fp).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs.append(img)
        if 'hha' in self.channels:
            hha_fp = os.path.join(self.root, 'data', imgname, 'hha_bfx.png')
            ahh = cv2.imread(hha_fp).astype(np.float32)         # cv2.read will change hha into ahh
            hha = cv2.cvtColor(ahh, cv2.COLOR_BGR2RGB)
            inputs.append(hha)
        if "depth" in self.channels:
            dep_fp = os.path.join(self.root, 'data', imgname, 'depth_bfx.png')
            dep = cv2.imread(dep_fp, cv2.IMREAD_UNCHANGED).astype(np.float32)
            dep = np.expand_dims(dep, axis=-1)
            inputs.append(dep)
        assert 0 < len(self.channels) == len(inputs),\
            "NYU Datasets input channels must be in ['rgb', 'hha', 'depth']"
        img = np.concatenate(inputs, axis=-1)

        mask_fp = os.path.join(self.root, 'data', imgname + 'label.png')
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
        mask -= 1       # 0->255

        image, mask = self.process(img, [mask])
        return image, mask.long()

    def __len__(self):
        return len(self.imglist)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll
