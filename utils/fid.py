from models.dit import DiffusionTransformerFlowModel

from pathlib import Path
from cleanfid import fid


def fid_guidance_sweep(
    dit: DiffusionTransformerFlowModel,
    root_dir,
    guidance_scales=(1.25, 1.5, 1.75, 2.0, 2.25),
    num_images=10_000,
    batch_size=256,
    num_timesteps=250,
    split="train",
    fid_batch_size=64,
):
    """Export samples and compute FID for a quick guidance-scale sweep."""
    scores = {}
    for scale in guidance_scales:
        out_dir = Path(root_dir) / f"w_{scale:g}"
        try:
            dit.sample(
                num_samples=num_images,
                out_dir=out_dir,
                batch_size=batch_size,
                guidance_scale=scale,
                num_timesteps=num_timesteps,
                overwrite=False,
            )
        except FileExistsError:
            print("already sampled")

        scores[scale] = fid.compute_fid(
            str(out_dir),
            dataset_name="cifar10",
            dataset_res=32,
            dataset_split=split,
            mode="clean",
            batch_size=fid_batch_size,
            device=next(dit.parameters()).device,
            use_dataparallel=False,
        )
    return scores
