from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NeuralCA(nn.Module):
    """A shared local update rule for image-growing cellular automata."""

    def __init__(
        self,
        channels: int = 16,
        hidden_size: int = 128,
        fire_rate: float = 0.5,
        perception: str = "sobel_laplace",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size
        self.fire_rate = fire_rate
        self.perception = perception

        filters = self._build_filters(perception)
        self.register_buffer("filters", filters)
        perceived_channels = channels * filters.shape[0]

        self.update_net = nn.Sequential(
            nn.Conv2d(perceived_channels, hidden_size, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

    @staticmethod
    def _build_filters(perception: str) -> torch.Tensor:
        identity = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.t()
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        )

        if perception == "identity":
            filters = [identity]
        elif perception == "sobel":
            filters = [identity, sobel_x, sobel_y]
        elif perception == "sobel_laplace":
            filters = [identity, sobel_x, sobel_y, laplace]
        else:
            raise ValueError(
                "perception must be one of: identity, sobel, sobel_laplace"
            )
        return torch.stack(filters).float()

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.shape[2] >= 3 and x.shape[3] >= 3

        filter_count = self.filters.shape[0]
        kernel = self.filters[:, None, :, :].repeat(self.channels, 1, 1, 1) # repeat for depth-wise convolution
        # ^ [:, None, :, :] same as .unsqueeze(1)
        y = F.conv2d(x, kernel, padding=1, groups=self.channels)
        batch, _, height, width = y.shape
        return y.view(batch, self.channels * filter_count, height, width)

    @staticmethod
    def living_mask(x: torch.Tensor) -> torch.Tensor:
        alpha = x[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 1,
        fire_rate: float | None = None,
    ) -> torch.Tensor:
        rate = self.fire_rate if fire_rate is None else fire_rate
        for _ in range(steps):
            pre_life = self.living_mask(x)
            dx = self.update_net(self.perceive(x))
            if rate < 1.0:
                update_mask = torch.rand(
                    x[:, :1].shape, dtype=x.dtype, device=x.device
                ) <= rate
                dx = dx * update_mask
            x = x + dx
            post_life = self.living_mask(x)
            x = x * (pre_life | post_life)
        return x

    def config(self) -> dict[str, object]:
        return {
            "channels": self.channels,
            "hidden_size": self.hidden_size,
            "fire_rate": self.fire_rate,
            "perception": self.perception,
        }

