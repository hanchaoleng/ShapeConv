import logging
import os
import random

import cv2
import numpy as np

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs, target_label):
        h, w, _ = inputs.shape
        th, tw = self.size
        x = int(round((w - tw) / 2.))
        y = int(round((h - th) / 2.))

        inputs = inputs[y: y + th, x: x + tw]
        target_label = target_label[y: y + th, x: x + tw]

        return inputs, target_label


class OfficialCrop(object):
    """
    [h_range=[45, 471], w_range=[41, 601]] -> (427, 561)
    official cropping to get best depth
    """
    def __call__(self, inputs, target_label):
        h, w, _ = inputs.shape
        assert h > 471 and w > 601, "inputs height must > 417, width > 601"
        inputs = inputs[45:471 + 1, 41:601 + 1]
        target_label = target_label[45:471 + 1, 41:601 + 1]
        return inputs, target_label


class DepthPredCrop(object):
    """
    640 * 480 -> dowmsample(320, 240) -> crop(304, 228) -> upsample(640, 480)
    """
    def __init__(self):
        self.center_crop = CenterCrop((228, 304))

    def __call__(self, inputs, target_label):
        inputs = cv2.resize(inputs, (320, 240), interpolation=cv2.INTER_LINEAR)
        target_label = cv2.resize(target_label, (320, 240), interpolation=cv2.INTER_NEAREST)

        inputs, target_label = self.center_crop(inputs, target_label)

        inputs = cv2.resize(inputs, (640, 480), interpolation=cv2.INTER_LINEAR)
        target_label = cv2.resize(target_label, (640, 480), interpolation=cv2.INTER_NEAREST)
        return inputs, target_label


@DATASETS.register_module
class NYUV2Dataset(BaseDataset):
    def __init__(self, root, imglist_name, classes=40, crop_paras=None, channels=None, transform=None,
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

        self.crop_process = None
        if crop_paras['type'] == "blank_crop":
            self.crop_process = CenterCrop(crop_paras['center_crop_size'])
        elif crop_paras['type'] == "official_crop":
            self.crop_process = OfficialCrop()
        elif crop_paras['type'] == "depth_pred_crop":
            self.crop_process = DepthPredCrop()

        logger.debug('Total of images is {}'.format(len(self.imglist)))
        self.classes = classes
        self.root = root
        self.channels = channels

    def read_inpus(self, imgname):
        inputs = []
        if 'rgb' in self.channels:
            img_fp = os.path.join(self.root, 'image', imgname + '.png')
            img = cv2.imread(img_fp).astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs.append(img)
        if 'hha' in self.channels:
            hha_fp = os.path.join(self.root, 'hha', imgname + '.png')
            ahh = cv2.imread(hha_fp).astype(np.float32)     # cv2.read will change hha into ahh
            hha = cv2.cvtColor(ahh, cv2.COLOR_BGR2RGB)
            inputs.append(hha)
        if "depth" in self.channels:
            dep_fp = os.path.join(self.root, 'depth', imgname + '.png')
            dep = cv2.imread(dep_fp, cv2.IMREAD_UNCHANGED).astype(np.float32)
            dep = np.expand_dims(dep, axis=-1)
            inputs.append(dep)
        assert 0 < len(self.channels) == len(inputs), \
            "NYU Dataset input channels must be in ['rgb', 'hha', 'depth']"
        img = np.concatenate(inputs, axis=-1)

        mask_fp = os.path.join(self.root, 'label' + str(self.classes), imgname + '.png')
        mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
        mask -= 1       # 0->255

        return img, mask

    def random_read(self):
        idx = random.randint(0, self.__len__()-1)
        imgname = self.imglist[idx]
        img, mask = self.read_inpus(imgname)
        return img, mask

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        img, mask = self.read_inpus(imgname)

        if self.crop_process:
            img, mask = self.crop_process(img, mask)

        img, mask = self.process(img, [mask])

        return img, mask.long()

    def __len__(self):
        return len(self.imglist)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll
