"""Plot learning curves from one or more experiment runs.

Reads metrics.jsonl files written by cs336_basics.logger.ExperimentLogger and
plots a chosen metric against both gradient steps and wall-clock time.

Example:
    uv run --with matplotlib python scripts/plot_curves.py \
        experiments/logs/run_a experiments/logs/run_b \
        --metric val_loss --out experiments/val_loss.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metric(run_dir: Path, metric: str):
    """Return (steps, times, values) for records in this run that have `metric`."""
    steps, times, values = [], [], []
    with (run_dir / "metrics.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if metric in rec:
                steps.append(rec["step"])
                times.append(rec["wall_clock_time"])
                values.append(rec[metric])
    return steps, times, values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Run directories (each with metrics.jsonl)")
    p.add_argument("--metric", default="val_loss")
    p.add_argument("--out", default="curves.png")
    args = p.parse_args()

    fig, (ax_steps, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    for run in args.runs:
        run_dir = Path(run)
        steps, times, values = load_metric(run_dir, args.metric)
        if not values:
            print(f"warning: no '{args.metric}' records in {run_dir}")
            continue
        label = run_dir.name
        ax_steps.plot(steps, values, marker=".", label=label)
        ax_time.plot(times, values, marker=".", label=label)

    ax_steps.set_xlabel("gradient steps")
    ax_steps.set_ylabel(args.metric)
    ax_steps.set_title(f"{args.metric} vs steps")
    ax_steps.grid(alpha=0.3)
    ax_steps.legend()

    ax_time.set_xlabel("wall-clock time (s)")
    ax_time.set_ylabel(args.metric)
    ax_time.set_title(f"{args.metric} vs wall-clock time")
    ax_time.grid(alpha=0.3)
    ax_time.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
