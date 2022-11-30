from utils.misc import is_dist_avail_and_initialized
import math

# Mostly copy from openmmlab
# https://github.com/open-mmlab/mmdetection/blob/e71b499608e9c3ccd4211e7c815fa20eeedf18a2/mmdet/core/hook/ema.py#L8
class BaseEMAHook(object):

    def __init__(self,
                 momentum=0.0002,
                 interval=1,
                 skip_buffers=False,
                 momentum_fun=None):

        assert 0 < momentum < 1
        self.momentum = momentum
        self.skip_buffers = skip_buffers
        self.interval = interval
        self.momentum_fun = momentum_fun

    def before_run(self, model):

        if is_dist_avail_and_initialized():
            model = model.module
        self.param_ema_buffer = {}
        if self.skip_buffers:
            self.model_parameters = dict(model.named_parameters())
        else:
            self.model_parameters = model.state_dict()
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())


    def get_momentum(self, _iter):
        return self.momentum_fun(_iter) if self.momentum_fun else self.momentum

    def after_train_iter(self, _iter):

        if (_iter + 1) % self.interval != 0:
            return
        momentum = self.get_momentum(_iter)
        for name, parameter in self.model_parameters.items():
            # exclude num_tracking
            if parameter.dtype.is_floating_point:
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(1 - momentum).add_(parameter.data, alpha=momentum)

    def after_train_epoch(self):
       
        self._swap_ema_parameters()

    def before_train_epoch(self):
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):

        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)



class ExpMomentumEMAHook(BaseEMAHook):
    """EMAHook using exponential momentum strategy.

    Args:
        total_iter (int): The total number of iterations of EMA momentum.
           Defaults to 2000.
    """

    def __init__(self, total_iter=2000, **kwargs):
        super(ExpMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: (1 - self.momentum) * math.exp(-(1 + x) / total_iter) + self.momentum


class LinearMomentumEMAHook(BaseEMAHook):
    """EMAHook using linear momentum strategy.

    Args:
        warm_up (int): During first warm_up steps, we may use smaller decay
            to update ema parameters more slowly. Defaults to 100.
    """

    def __init__(self, warm_up=100, **kwargs):
        super(LinearMomentumEMAHook, self).__init__(**kwargs)
        self.momentum_fun = lambda x: min(self.momentum**self.interval, (1 + x) / (warm_up + x))