from sacred.observers.base import RunObserver
from tensorboardX import SummaryWriter
import os


# Content types for `add_artifact`.
#
# The idea here is to walk through
# `https://tensorboardx.readthedocs.io/en/latest/tutorial.html#general-api-format`
# and make wrappers for each `add_`
#
# I am, however, afraid of ending up imitating `tensorboardX`'s interface
# with the use of `metadata` dict...

CONTENT_TYPES = dict()

def content_type(content_type):
    """@content_type(content_type)
    Takes `fun: Observer -> ArtifactName -> Array -> Metadata -> Iteration`
    and builds `new_fun: Observer -> ArtifactName -> Array -> Metadata`
    which is then registered in CONTENT_TYPES."""
    def register(fun):
        assert content_type not in CONTENT_TYPES
        def new_fun(self, name, filename, metadata):
            # iteration number is either maintained in `iter_numbers` of observer
            # or passed by the user in `metadata['iteration']`
            iteration = self.iter_numbers[content_type]
            if 'iteration' in metadata:
                iteration = metadata['iteration']
            self.iter_numbers[content_type] = max(self.iter_numbers[content_type] + 1, iteration)
            # extract the tensor for tensorboard
            array = numpy.read(filename)
            array = torch.from_numpy(array)
            # then call the underlying wrapper function
            fun(name, array, iteration)
        new_fun.__name__ = fun.__name__
        CONTENT_TYPES[content_type] = new_fun
        return new_fun
    return register

@content_type('numpy/image')
def tb_add_image(self, name, array, metadata, iteration):
    format_chw = len(array.shape) == 3 and array.shape[0] in [1, 3]
    format_hw = len(array.shape) == 2
    assert format_chw or format_hw
    fmt = 'CHW' if format_chw else 'HW'
    self.tensorboard.add_image(name, array, iteration, dataformats=fmt)

@content_type('numpy/hist')
def tb_add_hist(self, name, array, metadata, iteration):
    self.tensorboard.add_histogram(name, array, iteration)

# Not yet sure what to do with `add_figure`, probably it isn't enough
# to constrain ourselves to `ndarray`s.

@content_type('numpy/audio')
def tb_add_audio(self, name, array, metadata, iteration):
    assert 'sample_rate' in metadata
    sample_rate = metadata['sample_rate']
    self.tensorboard.add_audio(name, array, iteration, sample_rate)

@content_type('numpy/embedding')
def tb_add_embedding(self, name, array, metadata, iteration):
    # ignoring `name`
    self.tensorboard.add_embedding(array, metadata=metadata, label_img=, global_step=iteration)


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
        """Matches artifact's content_type and submits it to tensorboard in a proper way"""
        if content_type is None or content_type not in CONTENT_TYPES:
            return
        impl = CONTENT_TYPES[content_type]
        impl(self, name, filename, metadata)
