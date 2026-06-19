from typing import List, Type
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation: Type[torch.nn.Module] = torch.nn.SiLU,
        final_init: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

        if final_init:
            nn.init.zeros_(self.net[-1].weight)  # type: ignore
            nn.init.zeros_(self.net[-1].bias)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        Returns:
        - x: b n d
        """
        return self.net(x)


class MHA(nn.Module):
    """
    Multi-headed self-attention
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        Returns:
        - x: b n d
        """
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.out(x)
