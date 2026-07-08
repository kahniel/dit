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
        path: GaussianConditionalProbabilityPath,
        latent_stats: tuple[float, float] = (0.0, 1.0),
        null_ratio: float = 0.1,
    ):
        assert null_ratio > 0 and null_ratio < 1
        super().__init__(dataloader=dataloader, using_ema_model=True)

        self.path = path

        self.latent_mean, self.latent_std = latent_stats
        self.null_ratio = null_ratio

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
    def checkpoint(
        self,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        if ckpt_dir is None:
            ckpt_dir = self.output_dir
        state = {
            "raw": self.raw_model.state_dict(),
            "ema": self.model.state_dict(),
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
            grids = self.model.visualize_samples(
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
                    self.model,
                    f"samples/{self.run_name}_{ckpt_name}/",
                    num_images=1000,
                )
                for w, score in scores.items():
                    self.writer.add_scalar(f"train/fid_w_{w}", score, global_step)
                self.writer.flush()
