import statistics
import timeit
from enum import Enum
from typing import NamedTuple

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

VOCAB_SIZE = 10000
BATCH_SIZE = 2
CONTEXT_LENGTH = 512

WARMUP_STEPS = 5
MEASURE_STEPS = 10


# Define a structure for the architectural configurations
class ModelConfig(NamedTuple):
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


# Define the Enum mapping each size to its configuration
class ModelSize(Enum):
    SMALL = ModelConfig(768, 3072, 12, 12)
    MEDIUM = ModelConfig(1024, 4096, 24, 16)
    LARGE = ModelConfig(1280, 5120, 36, 20)
    # XL = ModelConfig(2560, 10240, 32, 32)
    # B10 = ModelConfig(4608, 12288, 50, 36)


def benchmark_full(model, optimizer, x_input, targets, device, warmup_steps, measure_steps):
    """Run warmup passes, then return per-step forward, backward, and optimizer
    times (seconds).

    The three phases are timed separately. We synchronize at each boundary so a
    given measurement captures actual kernel execution rather than just the async
    launch queue, and so one phase's time doesn't leak into the next.
    """
    def step():
        logits = model(x_input)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Warmup: let the allocator, cuDNN autotuner, and any lazy init settle.
    # Backward and the optimizer have their own kernels/state, so warm them too.
    # for _ in range(warmup_steps):
    #     step()
    if device == "cuda":
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    optimizer_times = []
    for _ in range(measure_steps):
        # --- Forward (including loss) ---
        start = timeit.default_timer()
        logits = model(x_input)
        loss = cross_entropy(logits, targets)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start)

        # --- Backward ---
        start = timeit.default_timer()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        backward_times.append(timeit.default_timer() - start)

        # --- Optimizer step ---
        start = timeit.default_timer()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        optimizer_times.append(timeit.default_timer() - start)

        # Zero grads immediately after the step so accumulation doesn't pollute
        # the next iteration (and is excluded from the timed regions above).
        optimizer.zero_grad(set_to_none=True)

    return forward_times, backward_times, optimizer_times


def _summarize(times):
    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0
    return mean_ms, std_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"batch={BATCH_SIZE}  ctx={CONTEXT_LENGTH}  "
          f"warmup={WARMUP_STEPS}  measure={MEASURE_STEPS}\n")

    for size in ModelSize:
        cfg = size.value
        model = None
        optimizer = None
        x_input = None
        targets = None
        try:
            model = BasicsTransformerLM(
                vocab_size=VOCAB_SIZE,
                context_length=CONTEXT_LENGTH,
                d_model=cfg.d_model,
                d_ff=cfg.d_ff,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
            ).to(device)

            x_input = torch.randint(
                low=0,
                high=VOCAB_SIZE,
                size=(BATCH_SIZE, CONTEXT_LENGTH),
                device=device,
            )
            targets = torch.randint(
                low=0,
                high=VOCAB_SIZE,
                size=(BATCH_SIZE, CONTEXT_LENGTH),
                device=device,
            )

            optimizer = AdamW(model.parameters())

            n_params = sum(p.numel() for p in model.parameters())

            forward_times, backward_times, optimizer_times = benchmark_full(
                model, optimizer, x_input, targets, device, WARMUP_STEPS, MEASURE_STEPS
            )
            fwd_mean, fwd_std = _summarize(forward_times)
            bwd_mean, bwd_std = _summarize(backward_times)
            opt_mean, opt_std = _summarize(optimizer_times)

            print(f"{size.name:>6} | {n_params / 1e6:7.1f}M params | "
                  f"fwd {fwd_mean:8.2f} ± {fwd_std:6.2f} ms | "
                  f"bwd {bwd_mean:8.2f} ± {bwd_std:6.2f} ms | "
                  f"opt {opt_mean:8.2f} ± {opt_std:6.2f} ms")

        except torch.cuda.OutOfMemoryError:
            print(f"{size.name:>6} | OOM - skipped")
        finally:
            del model
            del optimizer
            del x_input
            del targets
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
