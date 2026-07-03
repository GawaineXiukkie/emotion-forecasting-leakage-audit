"""
Shift-weighted losses (shift is the minority class). Binary: logit per position, target
in {0,1} or IGNORE_INDEX. All losses mask out IGNORE_INDEX positions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .dataset import IGNORE_INDEX


def _valid_mask(targets: torch.Tensor) -> torch.Tensor:
    return targets != IGNORE_INDEX


def class_balanced_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Weighted BCE-with-logits. pos_weight up-weights the shift (positive) class.
    A good default is pos_weight = (#neg / #pos) on the train fold."""
    mask = _valid_mask(targets)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    lg, tg = logits[mask], targets[mask].float()
    w = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(lg, tg, pos_weight=w)


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
              alpha: float = 0.75) -> torch.Tensor:
    """Binary focal loss. alpha weights the positive (shift) class."""
    mask = _valid_mask(targets)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    lg, tg = logits[mask], targets[mask].float()
    p = torch.sigmoid(lg)
    ce = F.binary_cross_entropy_with_logits(lg, tg, reduction="none")
    p_t = p * tg + (1 - p) * (1 - tg)
    alpha_t = alpha * tg + (1 - alpha) * (1 - tg)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()


def make_loss(name: str, pos_weight: float = 1.0):
    if name == "focal":
        return lambda lg, tg: focal_bce(lg, tg)
    if name == "cb_ce":
        return lambda lg, tg: class_balanced_bce(lg, tg, pos_weight)
    raise ValueError(f"unknown loss {name}")
