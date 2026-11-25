"""Losses. Class weights w_c = sqrt(n_N / n_c) (i.e. proportional to 1 / sqrt(n_c), normalised to N = 1),
computed on the training split only.
  wce        weighted cross-entropy
  focal1/15/2 focal loss, gamma 1 / 1.5 / 2, alpha_c = w_c
"""
import torch
import torch.nn.functional as F


def class_weights(y: torch.Tensor) -> torch.Tensor:
    n = torch.bincount(y, minlength=3).float().clamp(min=1)
    return torch.sqrt(n[0] / n)


def make_loss(name: str, w: torch.Tensor):
    if name == "wce":
        return lambda logits, y: F.cross_entropy(logits, y, weight=w)
    gamma = {"focal1": 1.0, "focal15": 1.5, "focal2": 2.0}[name]

    def focal(logits, y):
        lp = F.log_softmax(logits, 1).gather(1, y[:, None])[:, 0]
        return (-w[y] * (1 - lp.exp()) ** gamma * lp).sum() / w[y].sum()
    return focal
