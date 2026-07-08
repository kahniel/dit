from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

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
        self,
        data_channels: int,
        hidden_channels: list[int],
        beta: float = 0.1,
        reverse_transform=None,
    ):
        super().__init__()
        self.beta = beta
        self.reverse_transform = reverse_transform

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

    def get_stats(self, device: torch.device = torch.device("cpu")):
        dtype = next(self.parameters()).dtype
        latent_mean, latent_std = self.latent_stats
        latent_mean = torch.as_tensor(latent_mean, device=device, dtype=dtype)
        latent_std = torch.as_tensor(latent_std, device=device, dtype=dtype)
        return latent_mean, latent_std

    @torch.no_grad()
    def convert_to_latent_dataset(self, dataset, batch_size: int = 256):
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

    def get_preview_batch(self, dataloader, n=10):
        if not hasattr(self, "preview_batch"):
            batch = next(iter(dataloader))
            self.preview_batch = batch

        batch = self._move_to_device(
            self.preview_batch,
            next(self.model.parameters()).device,
        )
        x, y = batch
        return x[:n], y[:n]

    @torch.no_grad()
    def visualize_samples(
        self,
        save_path: Optional[str] = None,
        num_images=10,
        get_new=True,
        images =None,
        dataloader: Optional[DataLoader] = None,
        title: Optional[str] = None,
    ) -> torch.Tensor:
        if (images is None) == (dataloader is None):
            raise ValueError('Either images or dataloader argument is required')
        
        if images is not None:
            x = images.to(next(self.parameters()).device)
            num_images = len(images)
        elif not get_new:
            x, _ = self.get_preview_batch(dataloader, num_images)
        else:
            x, _ = next(iter(dataloader))
            x = x[:num_images]

        _, _, x_mean = self(x)

        x = x.cpu()
        x_mean = x_mean.cpu()

        x_all = torch.cat([x, x_mean], dim=0)

        if self.reverse_transform is not None:
            x_all = self.reverse_transform(x_all)
        x_all = torch.clamp(x_all, 0.0, 1.0)

        grid = make_grid(x_all, nrow=num_images, normalize=False)

        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        if title is not None:
            plt.title(title)

        if save_path is not None:
            plt.savefig(
                save_path,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

        return grid
