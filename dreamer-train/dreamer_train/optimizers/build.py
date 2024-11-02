import torch.optim as optim

from ..registry import Registry, build_module

OPTIMIZERS = Registry()


OPTIMIZERS.register_module(optim.Adadelta)
OPTIMIZERS.register_module(optim.Adagrad)
OPTIMIZERS.register_module(optim.Adam)
OPTIMIZERS.register_module(optim.AdamW)
OPTIMIZERS.register_module(optim.SparseAdam)
OPTIMIZERS.register_module(optim.Adamax)
OPTIMIZERS.register_module(optim.ASGD)
OPTIMIZERS.register_module(optim.SGD)
OPTIMIZERS.register_module(optim.Rprop)
OPTIMIZERS.register_module(optim.RMSprop)
OPTIMIZERS.register_module(optim.LBFGS)


def build_optimizer(params_or_type, *args, **kwargs):
    return build_module(OPTIMIZERS, params_or_type, *args, **kwargs)
