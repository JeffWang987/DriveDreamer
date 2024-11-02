import torch.nn as nn


class ModuleList(nn.ModuleList):
    def forward(self, idx, *args, **kwargs):
        return self[idx](*args, **kwargs)


class ModuleDict(nn.ModuleDict):
    def forward(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)
