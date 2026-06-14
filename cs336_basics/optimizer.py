"""Training optimizers built from scratch for the CS336 Transformer."""

import math

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Unlike plain Adam's L2 regularization (which folds lambda*theta into the
    gradient and lets it pass through the adaptive 1/sqrt(v) scaling), AdamW
    applies the decay -lr*lambda*theta directly to the parameter, decoupled from
    the adaptive moment update.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Lazy state init: step counter and first/second moment buffers.
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                # Update biased first and second moment estimates.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected step size folds both corrections into lr.
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Adaptive gradient step, then decoupled weight decay.
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Learning rate at step `it`: linear warmup, then cosine decay, then constant.

    - it < warmup: linear ramp 0 -> max_learning_rate.
    - warmup <= it <= cosine_cycle: cosine decay max -> min.
    - it > cosine_cycle: held at min_learning_rate.
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    return min_learning_rate
