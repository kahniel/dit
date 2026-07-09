import torch
import torch.nn as nn
from torchvision.utils import make_grid
from einops import rearrange
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from typing import Optional
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os

from models.components import MLP, MHA
from models.vae import VAE
from flow.ode import CFGVectorFieldODE, ConditionalVectorField, EulerSimulator


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()

        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: b
        Returns:
        - embeddings: b d
        """
        t = t.view(-1, 1)

        frequencies = 2 * torch.pi * t * self.weights

        cos_values = torch.cos(frequencies)
        sin_values = torch.sin(frequencies)

        return torch.cat([cos_values, sin_values], dim=-1)


class Patchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.c_in = c_in
        self.dim = dim

        self.proj = nn.Conv2d(self.c_in, self.dim, self.patch_size, self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, 1, img_size, img_size)
        Returns:
        - x: (bs, 1, img_size, img_size)
        """

        x = self.proj(x)
        x = rearrange(x, "b d h w -> b (h w) d")

        return x


class DiffusionTransformerLayer(nn.Module):
    def __init__(
        self,
        n_tokens,
        dim: int,
        heads: int,
    ):
        """
        Args:
        - n_tokens: sequence length (for sake of positional embeddings)
        - dim: dimension of hidden layers
        - heads: number of attention heads
        """
        super().__init__()

        self.n_tokens = n_tokens
        self.dim = dim
        self.heads = heads

        # Init normalization
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)

        # Initialize conditioning to zero - stabilizes residual connection!
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False), nn.Linear(dim, dim * 6)
        )
        nn.init.zeros_(self.ada_ln[1].weight)  # type: ignore
        nn.init.zeros_(self.ada_ln[1].bias)  # type: ignore

        # Init attention
        self.mha = MHA(dim, heads)

        # Init feedforward
        self.feedforward = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        - c: b d
        Returns:
        - x: b n d
        """
        # Compute conditioning gating, scaling, and bias
        c = rearrange(self.ada_ln(c), "b d -> b 1 d")
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(
            6, dim=-1
        )

        # Attention + residual connection
        x_norm1 = self.norm1(x)
        x_modulated1 = (1 + attn_scale) * x_norm1 + attn_bias

        attn_output = self.mha(x_modulated1)
        x = x + attn_output * attn_gate

        # Feedforward + residual connection
        x_norm2 = self.norm2(x)
        x_modulated2 = (1 + ff_scale) * x_norm2 + ff_bias

        ff_output = self.feedforward(x_modulated2)
        x = x + ff_output * ff_gate

        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, depth: int, n_tokens: int, dim: int, heads: int):
        """
        Args:
        - depth: number of layers
        - n_tokens: sequence length (for sake of positional embeddings)
        - dim: dimension of hidden layers
        - heads: number of attention heads
        """
        super().__init__()

        self.depth = depth
        self.n_tokens = n_tokens
        self.dim = dim
        self.heads = heads

        self.positional_encodings = nn.Parameter(torch.randn(n_tokens, dim))
        self.dit_layers = nn.ModuleList(
            [DiffusionTransformerLayer(n_tokens, dim, heads) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        - c: b d
        Returns:
        - x: b n d
        """
        x = x + self.positional_encodings.unsqueeze(0)
        for layer in self.dit_layers:
            x = layer(x, c)

        return x


class Depatchifier(nn.Module):
    def __init__(
        self, img_size: int, patch_size: int, dim: int, final_dim: int, c_out: int
    ):
        super().__init__()
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        h = w = img_size // patch_size

        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, 4 * dim, final_dim * patch_size**2]),
            # Depatchify
            Rearrange(
                "b (h w) (f ph pw) -> b f (h ph) (w pw)",
                h=h,
                w=w,
                f=final_dim,
                ph=patch_size,
                pw=patch_size,
            ),
            # Final convolution
            nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        Returns:
        - x: b 1 32 32
        """

        return self.net(x)


class DiffusionTransformerFlowModel(ConditionalVectorField):
    def __init__(
        self,
        vae: VAE,
        img_size: int = 32,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        dim: int = 256,
        heads: int = 4,
        final_dim: int = 10,
        n_classes: int = 11,
        null_label: int = 10,
    ):
        super().__init__()
        self.arch = {
            'vae': vae.get_arch(),
            'img_size': img_size,
            'patch_size': patch_size,
            'num_layers': num_layers,
            'c': c,
            'dim': dim,
            'heads': heads,
            'final_dim': final_dim,
            'n_classes': n_classes,
            'null_label': null_label,
        }
        
        self.vae = vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # 0. Construct time_embedder and y_embedder
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = nn.Embedding(num_embeddings=n_classes, embedding_dim=dim)

        # 1. Construct patchifier
        self.patchifier = Patchifier(
            img_size=img_size, patch_size=patch_size, c_in=c, dim=dim
        )

        # 2. Construct DiT
        n_tokens = (img_size // patch_size) ** 2
        self.dit = DiffusionTransformer(
            depth=num_layers,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )

        # 3. Construct de-patchifier
        self.depatchifier = Depatchifier(
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            final_dim=final_dim,
            c_out=c,
        )

        self.null_label = null_label
    
    def get_arch(self):
        return self.arch
    
    @classmethod
    def from_arch(cls, arch):
        arch['vae'] = VAE.from_arch(arch['vae'])
        return cls(**arch)
    
    @classmethod
    def from_ckpt(cls, ckpt_name: str, ckpt_dir: str):
        state = torch.load(
            os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"), map_location="cpu", weights_only=False
        )
        model = cls.from_arch(state['arch'])
        model.load_state_dict(state["model"])
        
        return model


    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
        - x: b 1 32 32
        - t: b
        - c: b
        Returns:
        - u_t^theta(x|y): b 1 32 32
        """
        # 1. Embed time and y
        t_emb, y_emb = self.time_embedder(t), self.y_embedder(y)

        # 2. Patchify
        x = self.patchifier(x)

        # 3. Pass through DiT
        x = self.dit(x, t_emb + y_emb)

        # 4. Depatchify
        x = self.depatchifier(x)

        return x


    def load(self, ckpt_name: str, ckpt_dir: str):
        state = torch.load(
            os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"), map_location="cpu", weights_only=False
        )

        self.load_state_dict(state["model"])
    
    @torch.no_grad()
    def checkpoint(
        self,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
    ):
        state = {"model": self.state_dict()}

        torch.save(state, os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"))


    def _latent_shape(self) -> tuple[int, int, int]:
        return self.patchifier.c_in, self.patchifier.img_size, self.patchifier.img_size

    def _decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        device = z.device
        latent_mean, latent_std = self.vae.get_stats(device)
        z = z * latent_std + latent_mean

        self.vae.to(device)
        x = self.vae.decode(z)
        if self.vae.reverse_transform is not None:
            x = self.vae.reverse_transform(x)
        return torch.clamp(x, 0.0, 1.0)

    def _generate_samples(
        self,
        y: torch.Tensor,
        guidance_scale: float = 1.5,
        num_timesteps: int = 250,
        use_tqdm: bool = False,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device

        y = y.to(device)
        latent_shape = self._latent_shape()
        z0 = torch.randn(y.shape[0], *latent_shape, device=device)

        ode = CFGVectorFieldODE(
            self,
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

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        out_dir,
        batch_size: int = 256,
        guidance_scale: float = 1.5,
        num_timesteps: int = 250,
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
        pbar = tqdm(total=num_samples, desc=f"generating samples w={guidance_scale}")
        while written < num_samples:
            cur_bs = min(batch_size, num_samples - written)
            device = next(self.parameters()).device
            y = (torch.arange(written, written + cur_bs, device=device) % 10).long()
            x = self._generate_samples(
                y=y,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
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
        num_timesteps: int = 250,
        guidance_scales: tuple[float, ...] = (1.0, 1.5, 2.0),
        use_tqdm: bool = False,
        title: Optional[str] = None,
    ) -> dict[float, torch.Tensor]:
        fig, axes = plt.subplots(
            1,
            len(guidance_scales),
            figsize=(4 * len(guidance_scales), 4),
            squeeze=False,
        )
        axes = axes[0]

        grids = {}
        for idx, guidance_scale in enumerate(guidance_scales):
            y = torch.arange(10, dtype=torch.int64).repeat_interleave(samples_per_class)
            x = self._generate_samples(
                y=y,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                use_tqdm=use_tqdm,
            )

            grid = make_grid(x, nrow=samples_per_class, normalize=False)
            axes[idx].imshow(grid.permute(1, 2, 0).cpu())
            axes[idx].axis("off")
            axes[idx].set_title(f"w={guidance_scale:g}")
            grids[guidance_scale] = grid

        if title is not None:
            fig.suptitle(title)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return grids
