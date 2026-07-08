from models import VAE
from flow import GaussianConditionalProbabilityPath, CFGVectorFieldODE, EulerSimulator
from trainers.trainer import Trainer
from utils.fid import fid_guidance_sweep

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader


class LatentCFGTrainer(Trainer):
    def __init__(
        self,
        dataloader: DataLoader,
        vae: VAE,
        path: GaussianConditionalProbabilityPath,
        latent_stats: tuple[float, float] = (0.0, 1.0),
        null_ratio: float = 0.1,
    ):
        assert null_ratio > 0 and null_ratio < 1
        super().__init__(dataloader=dataloader, using_ema_model=True)

        self.vae = vae
        self.path = path

        self.latent_mean, self.latent_std = latent_stats
        self.null_ratio = null_ratio

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

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
    def checkpoint(
        self,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        if ckpt_dir is None:
            ckpt_dir = self.output_dir
        state = {
            "raw": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "opt": self.opt.state_dict(),
            "global_step": global_step,
            "steps": self.steps,
            "losses": self.losses,
        }

        torch.save(state, os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"))

        title = f"Latent CFG samples ({ckpt_name})"
        if len(self.losses) > 0:
            title += f", loss={self.losses[-1]:.4f}"

        if global_step is not None:
            grids = self.ema_model.visualize_samples(
                save_path=os.path.join(ckpt_dir, f"{ckpt_name}_output.png"),
                title=title,
            )
            if (
                global_step is not None
                and hasattr(self, "writer")
                and self.writer is not None
            ):
                for guidance_scale, grid in grids.items():
                    self.writer.add_image(
                        f"samples/guidance_{guidance_scale:.1f}", grid, global_step
                    )
                    self.writer.flush()

                scores = fid_guidance_sweep(
                    self.ema_model,
                    f"samples/{self.run_name}_{ckpt_name}/",
                    num_images=1000,
                )
                for w, score in scores.items():
                    self.writer.add_scalar(f"train/fid_w_{w}", score, global_step)
                self.writer.flush()
