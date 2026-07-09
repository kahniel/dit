from models import VAE

from typing import Optional
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torchvision.utils import make_grid
from einops import rearrange
from torch.utils.data import DataLoader


@torch.no_grad()
def visualize_latent_interpolation(
    x1: torch.Tensor,
    x2: torch.Tensor,
    vae: VAE,
    n_steps: int,
    save_path: Optional[str] = None,
):
    vae.eval()

    z1_mean, z1_logvar = vae.encode(x1)
    z1 = z1_mean + torch.exp(0.5 * z1_logvar) * torch.randn_like(z1_mean)  # 1 c h w

    z2_mean, z2_logvar = vae.encode(x2)
    z2 = z2_mean + torch.exp(0.5 * z2_logvar) * torch.randn_like(z2_mean)  # 1 c h w

    lambdas = torch.linspace(0, 1, n_steps).to(z1.device)
    zs = (1 - lambdas) * z1.unsqueeze(-1) + lambdas * z2.unsqueeze(
        -1
    )  # 1 c h w n_steps
    zs = rearrange(zs, "1 c h w n -> n c h w")
    samples = vae.decode(zs)  # n_steps 1 h w

    if vae.reverse_transform is not None:
        samples = vae.reverse_transform(samples)

    grid = make_grid(samples, nrow=n_steps, normalize=True, value_range=(0, 1))
    plt.figure(figsize=(n_steps * 2, 2))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.title("Latent Interpolation")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    return samples
