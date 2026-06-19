import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import uuid
import random
import copy
import inspect

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

    def checkpoint(self, ckpt_name: str, global_step: int):
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

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
        run_name: str,
        resume_from: str,
    ):
        self.output_dir = os.path.join("runs/", run_name)

        self.model = model
        size_b = model_size_b(self.model)
        print(f"Loading model with size: {size_b / MiB:.3f} MiB")

        state = torch.load(
            f"{self.output_dir}/{resume_from}.pt",
            map_location="cpu",
        )

        self.model.load_state_dict(state["raw"])
        if self.using_ema_model:
            self.ema_model = copy.deepcopy(model).eval()
            self.ema_model.load_state_dict(state["ema"])

        self.losses = state["losses"]
        self.steps = state["steps"]
        self.losses_smoothed = state.get("losses_smoothed", [])

    def train(
        self,
        model: nn.Module,
        num_epochs: int,
        lr: float = 1e-3,
        run_name: Optional[str] = None,
        resume_from: Optional[str] = None,
        ema_decay: float = 0.999,
        warmup_steps: int = 0,
        log_every: int = 100,
        ckpt_every: Optional[int] = None,
        loss_smoothing: float = 0.99,
        **kwargs,
    ):
        """
        Linear warmup from 0 -> lr over `warmup_steps`, then constant lr.
        """
        # Initialize run name and output directory
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join("runs/", run_name)

        if resume_from is None:
            os.makedirs(self.output_dir, exist_ok=False)
            print("Initialized output directory at: " + self.output_dir)

        # Grab size
        if resume_from is not None:
            self.load(model, run_name, resume_from)
        else:
            self.model = model
            if self.using_ema_model:
                self.ema_model = copy.deepcopy(model).eval()

            size_b = model_size_b(self.model)
            print(f"Training model with size: {size_b / MiB:.3f} MiB")
            self.losses = []
            self.steps = []
            self.losses_smoothed = []

        # Initialize optimizer and LR
        self.opt = self.get_optimizer(lr)
        if resume_from is not None:
            state = torch.load(f"{self.output_dir}/{resume_from}.pt")
            self.opt.load_state_dict(state["opt"])

        self.model.train()
        device = next(self.model.parameters()).device

        global_step = self.steps[-1] if len(self.steps) > 0 else 0
        loss_smoothed = (
            self.losses_smoothed[-1] if len(self.losses_smoothed) > 0 else None
        )

        self._set_lr(0.0 if warmup_steps > 0 else lr)

        self.writer = SummaryWriter(
            log_dir=os.path.join(self.output_dir, "tensorboard")
        )
        pbar = tqdm(
            total=num_epochs * len(self.dataloader), desc=f"Epoch {1}/{num_epochs}"
        )
        for epoch in range(1, num_epochs + 1):
            for batch in self.dataloader:
                if warmup_steps > 0:
                    cur_lr = lr * min(1.0, (global_step + 1) / warmup_steps)
                    self._set_lr(cur_lr)
                else:
                    cur_lr = lr

                batch = self._move_to_device(batch, device)

                self.opt.zero_grad(set_to_none=True)
                loss = self._call_get_train_loss(batch, **kwargs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.update_ema(decay=ema_decay)

                global_step += 1
                loss_value = float(loss.detach().item())

                if loss_smoothed is None:
                    loss_smoothed = loss_value
                else:
                    loss_smoothed = (
                        loss_smoothing * loss_smoothed
                        + (1 - loss_smoothing) * loss_value
                    )

                if global_step % log_every == 0:
                    self.losses.append(loss_value)
                    self.losses_smoothed.append(loss_smoothed)
                    self.steps.append(global_step)
                    self.writer.add_scalar("train/loss", loss_value, global_step)
                    self.writer.add_scalar("train/lr", cur_lr, global_step)

                pbar.update()
                pbar.set_description(
                    f"Epoch {epoch}/{num_epochs}, step={global_step}, "
                    f"lr={cur_lr:.2e}, loss={loss_value:.4f}"
                )

            if ckpt_every is not None and epoch % ckpt_every == 0:
                self.model.eval()
                self.checkpoint(f"epoch_{epoch}", global_step)
                self.model.train()

        if ckpt_every is None or num_epochs % ckpt_every != 0:
            self.checkpoint(f"epoch_{num_epochs}", global_step)

        self.model.eval()

    @torch.no_grad()
    def update_ema(self, decay=0.999):
        if self.using_ema_model:
            for p, p_ema in zip(self.model.parameters(), self.ema_model.parameters()):
                p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)

            for b, b_ema in zip(self.model.buffers(), self.ema_model.buffers()):
                b_ema.copy_(b)
