"""CBAM for 1-D signals (Woo et al. 2018): channel attention (shared MLP on average- and max-pooled descriptors)
followed by temporal attention (k = 7 convolution on the channel-wise average and max maps)."""
import torch
import torch.nn as nn


class CBAM1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, kernel: int = 7):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, channels))
        self.temporal = nn.Conv1d(2, 1, kernel, padding=kernel // 2)
        self.last_channel = self.last_temporal = None

    def forward(self, x):                                      # (B, C, T)
        ca = torch.sigmoid(self.mlp(x.mean(-1)) + self.mlp(x.amax(-1)))[..., None]
        x = x * ca
        ta = torch.sigmoid(self.temporal(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        self.last_channel, self.last_temporal = ca.detach(), ta.detach()
        return x * ta
