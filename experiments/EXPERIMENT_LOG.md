# Experiment Log

A running record of training runs for the assignment. Each run is logged
automatically by `cs336_basics.logger.ExperimentLogger` to
`experiments/logs/<run_name>/` (`config.json` + `metrics.jsonl`).

## How to run & log an experiment

```sh
# Train (writes experiments/logs/<run-name>/{config.json,metrics.jsonl})
uv run scripts/train.py \
    --train-data data/ts_train.npy --val-data data/ts_valid.npy \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
    --batch-size 32 --max-iters 5000 --lr 3e-4 --device cuda \
    --run-name baseline --checkpoint-path checkpoints/baseline.pt

# Plot learning curves (val loss vs steps AND vs wall-clock time)
uv run --with matplotlib python scripts/plot_curves.py \
    experiments/logs/baseline --metric val_loss \
    --out experiments/baseline_val_loss.png

# Compare several runs on one figure
uv run --with matplotlib python scripts/plot_curves.py \
    experiments/logs/baseline experiments/logs/lr_1e-3 \
    --metric val_loss --out experiments/compare_lr.png
```

Each `metrics.jsonl` record carries `step`, `wall_clock_time`, and the logged
metrics, so curves can be drawn against gradient steps or wall-clock time.

## Summary table

| Run name | Changed from baseline | Final val loss | Steps | Wall-clock | Notes |
|----------|-----------------------|---------------:|------:|-----------:|-------|
| baseline | —                     | _TODO_         | 5000  | _TODO_     | reference config (~17M params) |
|          |                       |                |       |            |       |

## Runs

### baseline
- **Hypothesis / goal:** establish a reference TinyStories run.
- **Config:** d_model=512, layers=4, heads=16, d_ff=1344, ctx=256, lr=3e-4,
  warmup=200, batch=32, max_iters=5000. (full config in
  `experiments/logs/baseline/config.json`)
- **Result:** _final train/val loss, wall-clock time — TODO after running_
- **Curve:** `experiments/baseline_val_loss.png`
- **Takeaway:** _TODO_

<!--
Template for each subsequent entry — copy and fill in:

### <run-name>
- **Hypothesis / goal:** what question this run answers (e.g. "does lr=1e-3 train faster?").
- **Config / change:** only what differs from baseline.
- **Result:** final train/val loss, steps, wall-clock time.
- **Curve:** path to the saved plot.
- **Takeaway:** what you concluded; what to try next.
-->
