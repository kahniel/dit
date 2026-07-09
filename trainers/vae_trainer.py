from trainers.trainer import Trainer

import os
from typing import Optional
import torch
from torch.utils.data import DataLoader


class VAETrainer(Trainer):
    def __init__(
        self,
        dataloader: DataLoader,
        **kwargs,
    ):
        super().__init__(dataloader=dataloader)

    def get_train_loss(self, batch):
        x, y = batch
        z_mean, z_std, x_mean = self.model(x)
        return self.model.compute_loss(z_mean, z_std, x_mean, x)

    @torch.no_grad()
    def checkpoint(
        self,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        if ckpt_dir is None:
            ckpt_dir = self.output_dir
            
        self.model.init_latent_dataset(self.dataloader)
        state = {
            "model": self.model.state_dict(),
            "arch": self.model.get_arch(),
            "opt": self.opt.state_dict(),
            "global_step": global_step,
            "steps": self.steps,
            "losses": self.losses,
        }

        torch.save(state, os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"))
        grid = self.model.visualize_samples(
            save_path=os.path.join(ckpt_dir, f"{ckpt_name}_output.png"),
            get_new=False,
            title=f"VAE Reconstruction ({ckpt_name})",
        )

        if global_step is not None:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_image("samples/vae_reconstructions", grid, global_step)
                self.writer.flush()
