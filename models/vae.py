from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models.components import MHA, MLP


class ResidualBlock(nn.Module):
    """Two applications of LN + convolution + non-linearity + residual connection"""

    def __init__(self, channels: int, act=nn.SiLU):
        super().__init__()

        # Init norm, convolutions, and activations
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1), act()
        )

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1)

        # Initialize the second convolution to zero - stabilizes training early on!
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)  # type: ignore

    def forward(self, x: torch.Tensor):
        # Res init
        x_skip = x

        # Norm
        x = self.norm(x)

        # First convolution
        x = self.conv1(x)

        # Second convolution
        x = self.conv2(x)

        # Return residual connection
        return x_skip + x


class AttnBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 1):
        super().__init__()
        # Reshape
        self.reshape1 = Rearrange("b c h w -> b (h w) c")

        # Norm + attention
        self.norm1 = nn.LayerNorm(channels)
        self.mha = MHA(dim=channels, heads=heads)

        # Norm + ff
        self.norm2 = nn.LayerNorm(channels)
        self.ff = MLP([channels, 2 * channels, channels], final_init=True)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = self.reshape1(x)

        # Attention + residual connection
        x_skip = x
        x = self.norm1(x)
        x = x_skip + self.mha(x)

        # Feedforward + residual connection
        x_skip = x
        x = self.norm2(x)
        x = x_skip + self.ff(x)

        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, downsample_channels: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)

        self.downsample: nn.Module
        if downsample_channels is not None:
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=downsample_channels,
                padding=1,
                stride=2,
                kernel_size=3,
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        x = self.downsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list[int]):
        super().__init__()

        # Initial conv2d
        self.init_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            kernel_size=3,
            padding=1,
            stride=1,
        )

        # Initialize channels
        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        blocks = []
        for in_c, out_c in zip(ch_in, ch_out):
            blocks.append(EncoderBlock(in_c, out_c))

        self.blocks = nn.ModuleList(blocks)

        # Predict z_mean
        z_dim = hidden_channels[-1]
        self.z_mean = nn.Sequential(
            nn.GroupNorm(1, z_dim),
            nn.Conv2d(
                in_channels=z_dim,
                out_channels=z_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # Scalar log-variance
        self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)

        for block in self.blocks:
            x = block(x)

        z_mean = self.z_mean(x)

        return z_mean, self.logvar


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, upsample_channels: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)

        self.upsample: nn.Module
        if upsample_channels is not None:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=upsample_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
            )
        else:
            self.upsample = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        x = self.upsample(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: list[int]):
        super().__init__()

        # Initialize channels
        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        blocks = []
        for in_c, out_c in zip(ch_in, ch_out):
            blocks.append(DecoderBlock(in_c, out_c))

        self.blocks = nn.ModuleList(blocks)

        # Predict mean
        x_dim = hidden_channels[-1]
        self.x_mean = nn.Sequential(
            nn.GroupNorm(1, x_dim),
            nn.Conv2d(
                in_channels=x_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        x_mean = self.x_mean(x)

        return x_mean


class VAE(nn.Module):
    def __init__(
        self, data_channels: int, hidden_channels: list[int], beta: float = 0.1
    ):
        super().__init__()
        self.beta = beta

        # Encoder
        self._encoder = Encoder(data_channels, hidden_channels)

        # Decoder
        self._decoder = Decoder(data_channels, list(reversed(hidden_channels)))

    def encode(self, x: torch.Tensor):
        return self._encoder(x)

    def decode(self, z: torch.Tensor):
        return self._decoder(z)

    def forward(self, x: torch.Tensor):
        z_mean, z_logvar = self.encode(x)
        z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)
        x_mean = self.decode(z)
        return z_mean, z_logvar, x_mean

    def compute_loss(self, z_mean, z_logvar, x_mean, x_true):
        kl_loss = 0.5 * (z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1).mean()

        recon_loss = F.l1_loss(x_mean, x_true)

        return recon_loss + self.beta * kl_loss

    def set_stats(self, latent_stats: tuple):
        self.latent_stats = latent_stats

    def get_stats(self, device: torch.device = torch.cpu):
        dtype = next(self.parameters()).dtype
        latent_mean, latent_std = self.latent_stats
        latent_mean = torch.as_tensor(latent_mean, device=device, dtype=dtype)
        latent_std = torch.as_tensor(latent_std, device=device, dtype=dtype)
        return latent_mean, latent_std

    @torch.no_grad()
    def convert_to_latent_dataset(
        self, dataset, batch_size: int = 256, get_latent_stats=False
    ):
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        latents = []
        labels = []

        for x, y in tqdm(loader, desc="Converting dataset to latents"):
            x = x.to(device)
            z_mean, _ = self.encode(x)

            latents.append(z_mean.detach().cpu())
            labels.append(y.cpu())

        z = torch.cat(latents, dim=0)
        mean = z.mean(dim=(0, 2, 3), keepdim=True)
        std = z.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(1e-6)

        if was_training:
            self.train()

        self.set_stats((mean, std))
        return TensorDataset(torch.cat(latents), torch.cat(labels))
