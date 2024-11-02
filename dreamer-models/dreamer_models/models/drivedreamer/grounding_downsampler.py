import torch.nn as nn


class GroundingDownSampler(nn.Module):
    def __init__(self, in_dim=3, mid_dim=4, out_dim=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_dim, mid_dim, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_dim, out_dim, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)
