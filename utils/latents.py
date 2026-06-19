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
    reverse_transform=None,
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

    if reverse_transform is not None:
        samples = reverse_transform(samples)

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


@torch.no_grad()
def estimate_latent_stats(vae: VAE, dataloader: DataLoader, batches=100):
    zs = []
    vae.eval()
    device = next(vae.parameters()).device

    for _ in tqdm(range(1, batches + 1)):
        x, _ = next(iter(dataloader))
        x = x.to(device)
        z_mean, _ = vae.encode(x)
        zs.append(z_mean)

    z = torch.cat(zs, dim=0)
    mean = z.mean(dim=(0, 2, 3), keepdim=True)
    std = z.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return mean, std


@torch.no_grad()
def convert_to_latent_dataset(
    vae: VAE,
    dataset,
    batch_size: int = 256,
    latent_stats=None,
):
    device = next(vae.parameters()).device
    was_training = vae.training
    vae.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    labels = []

    for x, y in tqdm(loader, desc="Converting dataset to latents"):
        x = x.to(device)
        z_mean, _ = vae.encode(x)

        latents.append(z_mean.cpu())
        labels.append(y.cpu())

    if was_training:
        vae.train()

    return TensorDataset(torch.cat(latents), torch.cat(labels))
