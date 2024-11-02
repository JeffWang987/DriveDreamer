from torch.optim.lr_scheduler import LambdaLR

from ..registry import Registry, build_module

SCHEDULERS = Registry()


def build_scheduler(params_or_type, optimizer, *args, **kwargs):
    scheduler = build_module(SCHEDULERS, params_or_type, *args, **kwargs)
    ld_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=scheduler.get_value,
    )
    return ld_scheduler
