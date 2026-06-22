import statistics
import timeit
from enum import Enum
from typing import NamedTuple

import torch

from cs336_basics.model import BasicsTransformerLM

VOCAB_SIZE = 10000
BATCH_SIZE = 4
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


def benchmark_forward(model, x_input, device, warmup_steps, measure_steps):
    """Run warmup passes, then return a list of per-step forward times (seconds)."""
    # Warmup: let the allocator, cuDNN autotuner, and any lazy init settle.
    for _ in range(warmup_steps):
        model(x_input)
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure: synchronize inside the timed region so we capture actual
    # kernel execution, not just the async launch queue.
    times = []
    for _ in range(measure_steps):
        start = timeit.default_timer()
        model(x_input)
        if device == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    return times


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"batch={BATCH_SIZE}  ctx={CONTEXT_LENGTH}  "
          f"warmup={WARMUP_STEPS}  measure={MEASURE_STEPS}\n")

    for size in ModelSize:
        cfg = size.value
        model = None
        x_input = None
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

            n_params = sum(p.numel() for p in model.parameters())

            times = benchmark_forward(
                model, x_input, device, WARMUP_STEPS, MEASURE_STEPS
            )
            mean_ms = statistics.mean(times) * 1000
            std_ms = statistics.stdev(times) * 1000 if len(times) > 1 else 0.0

            print(f"{size.name:>6} | {n_params / 1e6:7.1f}M params | "
                  f"{mean_ms:8.2f} ms ± {std_ms:6.2f} ms")

        except torch.cuda.OutOfMemoryError:
            print(f"{size.name:>6} | OOM - skipped")
        finally:
            del model
            del x_input
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()