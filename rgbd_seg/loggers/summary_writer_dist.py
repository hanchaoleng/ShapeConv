from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class SummaryWriterDist():
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        self.writer = None
        if rank == 0:
            self.writer = SummaryWriter(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step, walltime)
