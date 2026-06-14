"""Train a Transformer language model.

Wires together the from-scratch components (model, AdamW, cosine LR schedule,
gradient clipping, data loader, checkpointing) into a single training loop.

Example:
    uv run scripts/train.py \
        --train-data data/ts_train.npy --val-data data/ts_valid.npy \
        --vocab-size 10000 --context-length 256 \
        --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
        --batch-size 32 --max-iters 5000 --device cuda \
        --checkpoint-path checkpoints/ts.pt
"""

import argparse

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.logger import ExperimentLogger
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.serialization import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    # Data: 1D token-id arrays on disk, loaded with np.memmap.
    p.add_argument("--train-data", type=str, required=True, help="Path to .npy token ids")
    p.add_argument("--val-data", type=str, default=None, help="Path to validation .npy token ids")

    # Model.
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)

    # Optimizer.
    p.add_argument("--lr", type=float, default=3e-4, help="Max (post-warmup) learning rate")
    p.add_argument("--min-lr", type=float, default=3e-5, help="Final cosine learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0, help="Max grad L2 norm (0 disables)")

    # LR schedule.
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--cosine-cycle-iters", type=int, default=None, help="Defaults to --max-iters")

    # Training.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # Logging / checkpointing.
    p.add_argument("--log-dir", type=str, default="experiments/logs")
    p.add_argument("--run-name", type=str, default=None, help="Defaults to a timestamp")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=50, help="Batches to average for val loss")
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    return p.parse_args()


def load_tokens(path: str) -> np.ndarray:
    """Memory-map a 1D .npy token-id array so we never load it all into RAM.

    np.load(mmap_mode="r") parses the .npy header (unlike np.memmap) and returns
    a read-only memory-mapped array, matching files written by Tokenizer.encode_file.
    """
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, context_length, device, eval_iters) -> float:
    """Average cross-entropy over `eval_iters` random batches (model in eval mode)."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        losses[i] = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return losses.mean().item()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cosine_cycle_iters = args.cosine_cycle_iters or args.max_iters

    train_data = load_tokens(args.train_data)
    val_data = load_tokens(args.val_data) if args.val_data else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params/1e6:.2f}M parameters; training on {args.device}")

    logger = ExperimentLogger(args.log_dir, args.run_name, config=vars(args))
    print(f"Logging to {logger.run_dir}")

    model.train()
    for it in range(start_iter, args.max_iters):
        # Set this step's learning rate from the cosine schedule.
        lr = get_lr_cosine_schedule(
            it, args.lr, args.min_lr, args.warmup_iters, cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # One optimization step.
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0:
            logger.log(it, {"train_loss": loss.item(), "lr": lr})

        if val_data is not None and args.eval_interval and it % args.eval_interval == 0 and it > 0:
            val_loss = estimate_loss(
                model, val_data, args.batch_size, args.context_length, args.device, args.eval_iters
            )
            logger.log(it, {"val_loss": val_loss})

        if (
            args.checkpoint_path
            and args.checkpoint_interval
            and it > 0
            and it % args.checkpoint_interval == 0
        ):
            save_checkpoint(model, optimizer, it, args.checkpoint_path)
            print(f"iter {it:6d} | saved checkpoint to {args.checkpoint_path}")

    # Final checkpoint.
    if args.checkpoint_path:
        save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)
        print(f"Done. Saved final checkpoint to {args.checkpoint_path}")

    logger.close()


if __name__ == "__main__":
    main()
