import logging
import os
import sys
import time

import torch.distributed as dist
from .summary_writer_dist import SummaryWriterDist


def build_logger(cfg, default_args):
    format_ = '%(asctime)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        pass

    for handler in cfg['handlers']:
        if handler['type'] == 'StreamHandler':
            instance = logging.StreamHandler(sys.stdout)
        elif handler['type'] == 'FileHandler':
            # only rank 0 will add a FileHandler
            if default_args.get('workdir') and rank == 0:
                fp = os.path.join(default_args['workdir'], 'logging.log')
                instance = logging.FileHandler(fp, 'w')
            else:
                continue
        else:
            instance = logging.StreamHandler(sys.stdout)

        level = getattr(logging, handler['level'])

        instance.setFormatter(formatter)
        if rank == 0:
            instance.setLevel(level)
        else:
            logger.setLevel(logging.ERROR)

        logger.addHandler(instance)

    return logger


def build_summarys_writer(cfg):
    writer = SummaryWriterDist(cfg.get('workdir'))
    return writer
