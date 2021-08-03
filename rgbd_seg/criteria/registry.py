import torch.nn as nn

from rgbd_seg.utils import Registry

CRITERIA = Registry('criterion')

CrossEntropyLoss = nn.CrossEntropyLoss
CRITERIA.register_module(CrossEntropyLoss)
