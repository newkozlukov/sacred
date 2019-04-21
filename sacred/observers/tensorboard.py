from tensorboardX import SummaryWriter
import os
import collections

from sacred.observers.base import RunObserver

class TensorboardObserver(RunObserver):
    def __init__(self, basedir):
        self.iter_numbers = collections.defaultdict(lambda: 0)
        self.basedir = os.path.abspath(basedir)

    def started_event(self, ex_info, command, host_info, start_time,
                      config, meta_info, _id):
        run_path = os.path.join(self.basedir, str(_id))
        assert not os.path.exists(run_path)
        self.run = {
                '_id': _id,
                'config': config,
                'start_time': start_time,
                'experiment': ex_info,
                'command': command,
                'host_info': host_info,
                }

        comment = (
                "♻ *{experiment[name]}* " \
                "started at _{start_time}_ " \
                "on host `{host_info[hostname]}`")
        if 'comment' in config:
            comment + ': {config[comment]}'
        comment = comment.format(**self.run)
        self.tensorboard = SummaryWriter(run_path, comment=comment)

    def log_metrics(self, metrics_by_name, info):
        """Store new measurements via tensorboardX.
        Reference: `https://github.com/IDSIA/sacred/blob/3333431079139f4bcb75616e6674bda89f3ac26d/sacred/observers/file_storage.py#L235`"""
        for name, metric_ptr in metrics_by_name.items():
            for val, it in zip(metric_ptr['values'], metric_ptr['steps']):
                self.tensorboard.add_scalar(name, val, it)

    def artifact_event(self, name, filename, metadata=None, content_type=None):
        # TODO:
        pass
