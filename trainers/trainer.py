import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import uuid
import random
import copy
import inspect
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MiB = 1024**2


def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(
        self,
        dataloader: DataLoader,
        using_ema_model: bool = False,
    ):
        super().__init__()
        self.dataloader = dataloader
        self.using_ema_model = using_ema_model

    @abstractmethod
    def get_train_loss(self, batch: Any, **kwargs) -> torch.Tensor:
        pass

    def _call_get_train_loss(self, batch: Any, **kwargs) -> torch.Tensor:
        signature = inspect.signature(self.get_train_loss)
        accepts_batch = "batch" in signature.parameters or any(
            p.kind is inspect.Parameter.VAR_POSITIONAL
            for p in signature.parameters.values()
        )
        if accepts_batch:
            return self.get_train_loss(batch, **kwargs)
        return self.get_train_loss(**kwargs)  # type: ignore[call-arg]

    def _move_to_device(self, batch: Any, device: torch.device) -> Any:
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=self.dataloader.pin_memory)
        if isinstance(batch, dict):
            return {
                key: self._move_to_device(value, device) for key, value in batch.items()
            }
        if isinstance(batch, tuple):
            return tuple(self._move_to_device(value, device) for value in batch)
        if isinstance(batch, list):
            return [self._move_to_device(value, device) for value in batch]
        return batch

    def _set_lr(self, lr: float):
        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def _get_lr(
        self,
        lr: float,
        global_step: int,
        total_steps: int,
        warmup_steps: int = 0,
        cosine: bool = False,
    ):
        if global_step < warmup_steps:
            cur_lr = lr * (global_step / warmup_steps)
        elif cosine:
            progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cur_lr = 0.1 * lr + 0.9 * lr * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            cur_lr = lr

        return cur_lr

    def checkpoint(
        self,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
        global_step: Optional[int] = None,
    ):
        pass

    def _checkpoint_name(self, epoch: int) -> str:
        return f"epoch_{epoch}_state"

    def _checkpoint_path(self, weights_dir: str, checkpoint_name: str) -> str:
        return os.path.join(weights_dir, f"{checkpoint_name}.pt")

    def get_optimizer(self, lr: float):
        if self.using_ema_model:
            return torch.optim.AdamW(
                (p for p in self.raw_model.parameters() if p.requires_grad),
                lr=lr,
                weight_decay=1e-4,
            )
        return torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=1e-4,
        )

    def random_name(self) -> str:
        adjectives = [
            "autumn",
            "hidden",
            "bitter",
            "misty",
            "silent",
            "empty",
            "dry",
            "dark",
            "summer",
            "icy",
            "delicate",
            "quiet",
            "white",
            "cool",
            "spring",
            "winter",
            "patient",
        ]
        foods = [
            "apple",
            "banana",
            "pear",
            "plum",
            "orange",
            "persimmon",
            "tangerine",
            "durian",
            "jackfruit",
            "jicama",
            "cantaloupe",
            "watermelon",
            "peach",
        ]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    def load(
        self,
        model,
        ckpt_name: str,
        ckpt_dir: Optional[str] = None,
    ):
        if ckpt_dir is None:
            ckpt_dir = self.output_dir

        state = torch.load(
            os.path.join(ckpt_dir, f"{ckpt_name}_state.pt"), map_location="cpu"
        )

        self.model = model
        size_b = model_size_b(self.model)
        print(f"Loading model with size: {size_b / MiB:.3f} MiB")
        if self.using_ema_model:
            self.raw_model = copy.deepcopy(self.model)
            self.raw_model.load_state_dict(state["raw"])
            self.model.load_state_dict(state["ema"])
        else:
            self.model.load_state_dict(state["raw"])

        if not hasattr(self, "lr"):
            self.lr = 1e-4
        self.opt = self.get_optimizer(self.lr)
        self.opt.load_state_dict(state["opt"])

        self.losses = state["losses"]
        self.steps = state["steps"]
        self.global_step = state.get(
            "global_step",
            self.steps[-1] if len(self.steps) > 0 else 0,
        )

    def train(
        self,
        model: nn.Module,
        num_epochs: int,
        lr: float = 1e-3,
        run_name: Optional[str] = None,
        resume_from: Optional[int] = None,
        ema_decay: float = 0.999,
        warmup_steps: int = 0,
        log_every: int = 100,
        ckpt_every: Optional[int] = None,
        cosine: bool = False,
        **kwargs,
    ):
        """
        Linear warmup from 0 -> lr over `warmup_steps`, then constant/cosine lr.
        `num_epochs` is the number of additional epochs to run. When resuming,
        `resume_from` is the last completed epoch number.
        """
        # Initialize run name and output directory
        run_name = run_name or self.random_name()
        self.run_name = run_name
        self.output_dir = os.path.join("runs/", run_name)
        start_epoch = resume_from or 0
        end_epoch = start_epoch + num_epochs

        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if resume_from is not None and resume_from < 0:
            raise ValueError("resume_from must be a non-negative epoch number")

        if resume_from is None:
            os.makedirs(self.output_dir, exist_ok=False)
            print("Initialized output directory at: " + self.output_dir)

        self.lr = lr
        # Grab size
        if resume_from is not None:
            self.load(model, self._checkpoint_name(resume_from))
            print(
                f"Resuming from epoch {resume_from}; training through epoch {end_epoch}"
            )
        else:
            self.model = model
            if self.using_ema_model:
                self.raw_model = copy.deepcopy(model)
            self.opt = self.get_optimizer(self.lr)

            size_b = model_size_b(self.model)
            print(f"Training model with size: {size_b / MiB:.3f} MiB")
            self.losses = []
            self.steps = []
            self.global_step = 0

        self.train_model = self.raw_model if self.using_ema_model else self.model
        self.train_model.train()
        device = next(self.train_model.parameters()).device

        global_step = self.global_step

        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch <= 0:
            raise ValueError("dataloader must contain at least one batch")
        total_steps = num_epochs * steps_per_epoch
        started_at = time.perf_counter()
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.output_dir, "tensorboard")
        )

        pbar = tqdm(total=total_steps, desc=f"Epoch {start_epoch + 1}/{end_epoch}")
        for epoch in range(start_epoch + 1, end_epoch + 1):
            for batch_idx, batch in enumerate(self.dataloader):
                cur_lr = self._get_lr(
                    lr, global_step, total_steps, warmup_steps, cosine
                )
                self._set_lr(cur_lr)

                batch = self._move_to_device(batch, device)

                self.opt.zero_grad(set_to_none=True)
                loss = self._call_get_train_loss(batch, **kwargs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)
                self.opt.step()
                self.update_ema(decay=ema_decay)

                global_step += 1
                loss_value = float(loss.detach().item())

                session_step = (
                    (epoch - start_epoch - 1) * steps_per_epoch + batch_idx + 1
                )
                elapsed_s = time.perf_counter() - started_at
                steps_per_sec = session_step / max(elapsed_s, 1e-9)
                remaining_steps = total_steps - session_step
                eta_s = remaining_steps / max(steps_per_sec, 1e-9)
                progress_pct = 100.0 * session_step / total_steps

                if global_step % log_every == 0 or batch_idx == steps_per_epoch - 1:
                    self.losses.append(loss_value)
                    self.steps.append(global_step)
                    self.writer.add_scalar("train/loss", loss_value, global_step)
                    self.writer.add_scalar(
                        "train/progress_pct", progress_pct, global_step
                    )
                    self.writer.add_scalar("train/eta_hours", eta_s / 3600, global_step)
                    self.writer.add_scalar(
                        "train/elapsed_hours", elapsed_s / 3600, global_step
                    )
                    self.writer.add_scalar(
                        "train/steps_per_sec", steps_per_sec, global_step
                    )
                    self.writer.add_scalar("train/epoch", epoch, global_step)
                    self.writer.add_scalar(
                        "train/session_step", session_step, global_step
                    )
                    self.writer.add_scalar(
                        "train/remaining_steps", remaining_steps, global_step
                    )
                    self.writer.flush()

                pbar.update()
                pbar.set_description(
                    f"Epoch {epoch}/{end_epoch}, step={global_step}, "
                    f"lr={cur_lr:.2e}, loss={loss_value:.4f}"
                )

            if ckpt_every is not None and epoch % ckpt_every == 0:
                self.model.eval()
                self.checkpoint(f"epoch_{epoch}", global_step=global_step)
                self.model.train()

        if ckpt_every is None or end_epoch % ckpt_every != 0:
            self.checkpoint(f"epoch_{end_epoch}", global_step=global_step)

        self.train_model.eval()
        pbar.close()
        self.writer.close()

    @torch.no_grad()
    def update_ema(self, decay=0.999):
        if self.using_ema_model:
            for p, p_ema in zip(self.raw_model.parameters(), self.model.parameters()):
                p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)

            for b, b_ema in zip(self.model.buffers(), self.model.buffers()):
                b_ema.copy_(b)
