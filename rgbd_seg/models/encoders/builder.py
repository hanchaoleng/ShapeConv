import torch.nn as nn

from rgbd_seg.utils import build_from_cfg
from .backbones.registry import BACKBONES
from .enhance_modules.registry import ENHANCE_MODULES


def build_encoder(cfg, default_args=None):
    backbone = build_from_cfg(cfg['backbone'], BACKBONES, default_args)

    enhance_cfg = cfg.get('enhance')
    # enhance_low_cfg = cfg.get('enhance_low')
    if enhance_cfg:
        enhance_module = build_from_cfg(enhance_cfg, ENHANCE_MODULES,
                                        default_args)
        # enhance_low_module = build_from_cfg(enhance_low_cfg, ENHANCE_MODULES,
        #                                     default_args)
        encoder = nn.Sequential(backbone, enhance_module)
        # encoder = nn.Sequential(backbone, enhance_module, enhance_low_module)
    else:
        encoder = backbone

    return encoder
