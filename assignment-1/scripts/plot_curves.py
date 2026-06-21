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
import time
from pathlib import Path

import matplotlib.pyplot as plt


def load_metric(run_dir: Path, metric: str):
    """Split metrics.jsonl into one segment per training run.

    The logger appends to metrics.jsonl, so re-running (or resuming) a run name
    concatenates multiple runs in one file. A new run is detected wherever `step`
    resets (decreases), so each is plotted as its own line instead of being
    joined by a spurious connector segment.

    Returns a list of dicts (one per run): {"steps", "times", "values", "start"},
    where "start" is the run's first absolute timestamp (or None for older logs
    written before timestamps were recorded).
    """
    segments = []
    current = None
    prev_step = None
    with (run_dir / "metrics.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if metric not in rec:
                continue
            step = rec["step"]
            if prev_step is None or step < prev_step:  # first record or run reset
                current = {"steps": [], "times": [], "values": [], "start": rec.get("timestamp")}
                segments.append(current)
            current["steps"].append(step)
            current["times"].append(rec["wall_clock_time"])
            current["values"].append(rec[metric])
            prev_step = step
    return segments


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Run directories (each with metrics.jsonl)")
    p.add_argument("--metric", default="val_loss")
    p.add_argument("--out", default="curves.png")
    args = p.parse_args()

    fig, (ax_steps, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
    for run in args.runs:
        run_dir = Path(run)
        segments = load_metric(run_dir, args.metric)
        if not segments:
            print(f"warning: no '{args.metric}' records in {run_dir}")
            continue
        for i, seg in enumerate(segments, start=1):
            label = run_dir.name
            if seg["start"] is not None:  # tag with the run's start time when available
                label += time.strftime(" %m-%d %H:%M", time.localtime(seg["start"]))
            elif len(segments) > 1:  # older logs without timestamps: fall back to #N
                label += f" #{i}"
            ax_steps.plot(seg["steps"], seg["values"], marker=".", label=label)
            ax_time.plot(seg["times"], seg["values"], marker=".", label=label)

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
