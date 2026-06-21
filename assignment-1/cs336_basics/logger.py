"""Lightweight experiment-tracking infrastructure.

Records metrics (e.g. train/val loss, learning rate) against both gradient steps
and wall-clock time, so learning curves can later be plotted on either x-axis.
Each run gets its own directory containing:
  - config.json    the run's hyperparameters
  - metrics.jsonl  one JSON record per logged step

This is intentionally dependency-free (no W&B); plot_curves.py reads the JSONL.
"""

import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        log_dir: str | Path,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        # Each run gets its own timestamped subdir under the run name, so
        # re-running the same name never clobbers a previous run's files:
        #   <log_dir>/<run_name>/<YYYYMMDD_HHMMSS>/{config.json,metrics.jsonl}
        run_name = run_name or "run"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_root = Path(log_dir) / run_name
        self.run_dir = run_root / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Point <run_name>/latest at this run so downstream tools (plot_curves,
        # generate) can use a stable path instead of the timestamp.
        latest = run_root / "latest"
        latest.unlink(missing_ok=True)
        latest.symlink_to(timestamp)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = self.metrics_path.open("a")
        self._start_time = time.time()

        if config is not None:
            with (self.run_dir / "config.json").open("w") as f:
                json.dump(config, f, indent=2, default=str)

    def log(self, step: int, metrics: dict[str, float], to_console: bool = True) -> None:
        """Append one record stamping `metrics` with the step and elapsed seconds."""
        now = time.time()
        wall_clock = now - self._start_time
        # Absolute timestamp too, so plots can split/label appended runs by time.
        record = {"step": step, "wall_clock_time": wall_clock, "timestamp": now, **metrics}

        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()

        if to_console:
            body = " | ".join(f"{k} {v:.4g}" for k, v in metrics.items())
            print(f"iter {step:6d} | {body} | {wall_clock:.1f}s")

    def close(self) -> None:
        self._metrics_file.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
