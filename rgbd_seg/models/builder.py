import torch.nn as nn


# from .decoders import build_brick
from rgbd_seg.models.decoders import build_brick
from rgbd_seg.models.decoders import build_decoder
from rgbd_seg.models.encoders import build_encoder
from rgbd_seg.models.heads import build_head


def build_model(cfg):
    encoder = build_encoder(cfg.get('encoder'))

    if cfg.get('decoder'):
        middle = build_decoder(cfg.get('decoder'))
        assert 'collect' not in cfg
    else:
        assert 'collect' in cfg
        middle = build_brick(cfg.get('collect'))

    head = build_head(cfg['head'])

    model = nn.Sequential(encoder, middle, head)

    return model
