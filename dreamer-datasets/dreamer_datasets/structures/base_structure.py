import torch


class BaseStructure:
    def __init__(self, tensor, **kwargs):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
        tensor = torch.as_tensor(tensor, device=device)
        self.tensor = tensor
        self.default_params = kwargs

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    def to(self, *args, **kwargs):
        return type(self)(self.tensor.to(*args, **kwargs), **self.default_params)

    def clone(self):
        return type(self)(self.tensor.clone(), **self.default_params)

    def contiguous(self):
        return type(self)(self.tensor.contiguous(), **self.default_params)

    def cuda(self, device):
        return type(self)(self.tensor.cuda(device), **self.default_params)

    def cpu(self):
        return type(self)(self.tensor.contiguous().cpu(), **self.default_params)

    def numpy(self):
        return self.tensor.contiguous().cpu().numpy()

    def new_structure(self, data):
        if not isinstance(data, torch.Tensor):
            new_tensor = self.tensor.new_tensor(data)
        else:
            new_tensor = data.to(self.device)
        return type(self)(new_tensor, **self.default_params)

    def __getitem__(self, item):
        new_tensor = self.tensor[item]
        if new_tensor.ndim == 1:
            new_tensor = new_tensor.view(1, -1)
        return type(self)(new_tensor, **self.default_params)

    def __iter__(self):
        yield from self.tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __repr__(self):
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'
