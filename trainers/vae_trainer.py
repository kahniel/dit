from trainers.trainer import Trainer
import os
from typing import Optional

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


class VAETrainer(Trainer):
    def __init__(
        self,
        dataloader: DataLoader,
        reverse_transform=None,
        **kwargs,
    ):
        super().__init__(dataloader=dataloader)
        self.reverse_transform = reverse_transform

    def get_train_loss(self, batch):
        x, y = batch
        z_mean, z_std, x_mean = self.model(x)
        return self.model.compute_loss(z_mean, z_std, x_mean, x)

    def get_preview_batch(self, n=10):
        if not hasattr(self, "preview_batch"):
            batch = next(iter(self.dataloader))
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
        images=None,
        title: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        if images is not None:
            x = self._move_to_device(
                images,
                next(self.model.parameters()).device,
            )
            num_images = len(images)
        elif not get_new:
            x, _ = self.get_preview_batch(num_images)
        else:
            x, _ = next(iter(self.dataloader))
            x = x[:num_images]
        
        _, _, x_mean = self.model(x)

        x = x.cpu()
        x_mean = x_mean.cpu()

        x_all = torch.cat([x, x_mean], dim=0)

        if self.reverse_transform is not None:
            x_all = self.reverse_transform(x_all)
        x_all = torch.clamp(x_all, 0.0, 1.0)

        grid = make_grid(x_all, nrow=num_images, normalize=False)

        if global_step is not None:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_image("samples/vae_reconstructions", grid, global_step)
                self.writer.flush()

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
            "opt": self.opt.state_dict(),
            "global_step": global_step,
            "steps": self.steps,
            "losses": self.losses,
        }

        torch.save(state, os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"))
        self.visualize_samples(
            save_path=os.path.join(ckpt_dir, f"{ckpt_name}_output.png"),
            get_new=False,
            title=f"VAE Reconstruction ({ckpt_name})",
            global_step=global_step,
        )
