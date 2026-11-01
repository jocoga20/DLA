import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        warmup_start_lr=1e-8,
        eta_min=1e-8,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # Warmup lineare
        if step < self.warmup_steps:
            alpha = step / max(1, self.warmup_steps - 1)

            return [
                self.warmup_start_lr
                + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        # Cosine decay
        progress = (
            step - self.warmup_steps
        ) / max(1, self.total_steps - self.warmup_steps)

        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]