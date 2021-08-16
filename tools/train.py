import os
import shutil
import sys
import argparse
import time
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from rgbd_seg.runners import TrainRunner
from rgbd_seg.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--distribute', default=False, action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--timestamp', type=str, default="yymmdd_hhmmss")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    _, fullname = os.path.split(cfg_path)
    fname, ext = os.path.splitext(fullname)

    root_workdir = cfg.pop('root_workdir')

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        timestamp = args.timestamp
    else:
        rank = 0
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    workdir = os.path.join(root_workdir, "%s_%s" % (fname, timestamp))
    if rank == 0:
        os.makedirs(workdir, exist_ok=True)
        shutil.copy(cfg_path, os.path.join(workdir, fullname))

    train_cfg = cfg['train']
    inference_cfg = cfg['inference']
    common_cfg = cfg['common']
    common_cfg['workdir'] = workdir
    common_cfg['distribute'] = args.distribute

    runner = TrainRunner(train_cfg, inference_cfg, common_cfg)
    runner()


if __name__ == '__main__':
    main()
