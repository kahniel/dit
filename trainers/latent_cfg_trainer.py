from models import VAE
from flow import GaussianConditionalProbabilityPath, CFGVectorFieldODE, EulerSimulator
from trainers.trainer import Trainer

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


class LatentCFGTrainer(Trainer):
    def __init__(
        self,
        dataloader: DataLoader,
        vae: VAE,
        path: GaussianConditionalProbabilityPath,
        null_label: int,
        latent_stats: tuple[float, float] = (0.0, 1.0),
        null_ratio: float = 0.1,
        eps: float = 0.001,
        reverse_transform=None,
    ):
        assert null_ratio > 0 and null_ratio < 1
        super().__init__(dataloader=dataloader, using_ema_model=True)

        self.vae = vae
        self.path = path

        self.null_label = null_label
        self.latent_mean, self.latent_std = latent_stats
        self.null_ratio = null_ratio
        self.reverse_transform = reverse_transform

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def _latent_stats(self, device: torch.device):
        dtype = next(self.vae.parameters()).dtype
        latent_mean = torch.as_tensor(self.latent_mean, device=device, dtype=dtype)
        latent_std = torch.as_tensor(self.latent_std, device=device, dtype=dtype)
        return latent_mean, latent_std

    def _latent_shape(self) -> tuple[int, int, int]:
        patchifier = getattr(self.model, "patchifier", None)
        if patchifier is not None:
            return patchifier.c_in, patchifier.img_size, patchifier.img_size

        batch = next(iter(self.dataloader))
        z, _ = batch
        return tuple(z.shape[1:])

    def _decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        latent_mean, latent_std = self._latent_stats(device)
        z = z * latent_std + latent_mean

        self.vae.to(device)
        x = self.vae.decode(z)
        if self.reverse_transform is not None:
            x = self.reverse_transform(x)
        return torch.clamp(x, 0.0, 1.0)

    def _generate_samples(
        self,
        y: torch.Tensor,
        guidance_scale: float = 1.5,
        num_timesteps: int = 250,
        use_raw: bool = False,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        sample_model = self.model if use_raw else self.ema_model
        sample_model.eval()
        device = next(sample_model.parameters()).device

        y = y.to(device)
        latent_shape = self._latent_shape()
        z0 = torch.randn(y.shape[0], *latent_shape, device=device)

        ode = CFGVectorFieldODE(
            sample_model,
            null_label=self.null_label,
            y=y,
            guidance_scale=guidance_scale,
        )
        simulator = EulerSimulator(ode)

        ts = (
            torch.linspace(0, 0.999, num_timesteps, device=device)
            .view(1, -1)
            .expand(y.shape[0], -1)
        )
        z1 = simulator.simulate(z0, ts, use_tqdm=use_tqdm)
        return self._decode_latents(z1)

    def get_train_loss(self, batch):
        z, y = batch
        latent_mean, latent_std = self._latent_stats(z.device)
        z_enc = (z - latent_mean) / latent_std

        batch_size = z.shape[0]

        mask = torch.rand(batch_size, device=y.device) < self.null_ratio
        y = torch.where(mask, self.null_label, y)

        t = torch.rand(batch_size, device=z_enc.device) * 0.999
        x = self.path.sample_conditional_path(z_enc, t)

        u_target = self.path.conditional_vector_field(x, z_enc, t)
        u_theta = self.model(x, t, y)

        return torch.nn.functional.mse_loss(u_theta, u_target)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        out_dir,
        batch_size: int = 256,
        guidance_scale: float = 1.5,
        num_timesteps: int = 250,
        use_raw=False,
        seed=None,
        overwrite=False,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        existing = sorted(out_path.glob("*.png"))

        if overwrite:
            for image_path in existing:
                image_path.unlink()
            existing = []
        elif existing:
            raise FileExistsError(
                f"{out_path} already contains PNG files. Use overwrite=True or a new directory."
            )

        written = 0
        pbar = tqdm(total=num_samples, desc=f"export FID samples w={guidance_scale}")
        while written < num_samples:
            cur_bs = min(batch_size, num_samples - written)
            device = next(
                (self.model if use_raw else self.ema_model).parameters()
            ).device
            y = (torch.arange(written, written + cur_bs, device=device) % 10).long()
            x = self._generate_samples(
                y=y,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                use_raw=use_raw,
                use_tqdm=False,
            )

            x_uint8 = (
                (x * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            )
            for img in x_uint8:
                Image.fromarray(img).save(out_path / f"{written:06d}.png")
                written += 1

            pbar.update(cur_bs)
        pbar.close()

    @torch.no_grad()
    def visualize_samples(
        self,
        save_path: Optional[str] = None,
        samples_per_class: int = 5,
        num_timesteps: int = 100,
        guidance_scales: tuple[float, ...] = (1.0, 1.5, 2.0),
        use_tqdm: bool = False,
        use_raw: bool = False,
        title: Optional[str] = None,
    ):
        fig, axes = plt.subplots(
            1,
            len(guidance_scales),
            figsize=(4 * len(guidance_scales), 4),
            squeeze=False,
        )
        axes = axes[0]

        for idx, guidance_scale in enumerate(guidance_scales):
            y = torch.arange(10, dtype=torch.int64).repeat_interleave(samples_per_class)
            x = self._generate_samples(
                y=y,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                use_raw=use_raw,
                use_tqdm=use_tqdm,
            )

            grid = make_grid(x, nrow=samples_per_class, normalize=False)
            axes[idx].imshow(grid.permute(1, 2, 0).cpu())
            axes[idx].axis("off")
            axes[idx].set_title(f"w={guidance_scale:.1f}")

        if title is not None:
            fig.suptitle(title)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    @torch.no_grad()
    def checkpoint(self, ckpt_name: str):
        state = {
            "raw": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "opt": self.opt.state_dict(),
            "steps": self.steps,
            "losses": self.losses,
            "losses_smoothed": self.losses_smoothed,
        }

        torch.save(state, os.path.join(self.output_dir, f"{ckpt_name}_state.pt"))

        if len(self.steps) > 0:
            plt.figure()
            plt.plot(self.steps, self.losses_smoothed)
            plt.xlabel("Step")
            plt.ylabel("Smoothed loss")
            plt.title("Training Loss")
            plt.savefig(
                os.path.join(self.output_dir, f"{ckpt_name}_lossplot.png"),
                bbox_inches="tight",
            )
            plt.close()

        title = f"Latent CFG samples ({ckpt_name})"
        if len(self.losses_smoothed) > 0:
            title += f", loss={self.losses_smoothed[-1]:.4f}"

        self.visualize_samples(
            save_path=os.path.join(self.output_dir, f"{ckpt_name}_output.png"),
            title=title,
        )
