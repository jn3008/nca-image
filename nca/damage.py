from __future__ import annotations

import math
import random

import torch


def erase_rectangles(x: torch.Tensor, fraction: float, count: int = 1) -> torch.Tensor:
    damaged = x.clone()
    _, _, height, width = damaged.shape
    area = height * width * fraction / max(count, 1)
    side = max(1, int(math.sqrt(area)))
    for batch in range(damaged.shape[0]):
        for _ in range(count):
            rect_h = random.randint(max(1, side // 2), max(1, side * 2))
            rect_w = max(1, int(area / rect_h))
            top = random.randint(0, max(0, height - rect_h))
            left = random.randint(0, max(0, width - rect_w))
            damaged[batch, :, top : top + rect_h, left : left + rect_w] = 0.0
    return damaged


def erase_circles(x: torch.Tensor, fraction: float, count: int = 1) -> torch.Tensor:
    damaged = x.clone()
    _, _, height, width = damaged.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device),
        torch.arange(width, device=x.device),
        indexing="ij",
    )
    radius = max(1, int(math.sqrt(height * width * fraction / (math.pi * count))))
    for batch in range(damaged.shape[0]):
        for _ in range(count):
            cy = random.randint(0, height - 1)
            cx = random.randint(0, width - 1)
            mask = (yy - cy).square() + (xx - cx).square() <= radius * radius
            damaged[batch, :, mask] = 0.0
    return damaged


def random_damage(x: torch.Tensor, fraction: float, shape: str = "circle") -> torch.Tensor:
    if fraction <= 0:
        return x
    if shape == "circle":
        return erase_circles(x, fraction=fraction)
    if shape == "rectangle":
        return erase_rectangles(x, fraction=fraction)
    raise ValueError("shape must be 'circle' or 'rectangle'")

