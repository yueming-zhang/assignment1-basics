# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training and generation

The full pipeline is: **train a BPE tokenizer → tokenize the corpus to `.npy` →
train the model → generate text**. Steps 1–2 only need to be done once per dataset.

### 0. Prerequisite: tokenizer + tokenized data

Train a BPE tokenizer (TinyStories, 10K vocab) and encode the splits to
memory-mappable `uint16` `.npy` token arrays:

```sh
# Train the BPE tokenizer -> scripts/ts_bpe_output/{vocab.json,merges.json}
uv run python scripts/train_tokenizer.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 --out-dir scripts/ts_bpe_output

# Encode train/valid text -> .npy token arrays
uv run scripts/tokenize_dataset.py \
    --vocab scripts/ts_bpe_output/vocab.json \
    --merges scripts/ts_bpe_output/merges.json \
    --input data/TinyStoriesV2-GPT4-train.txt --output data/tokenized/ts_train.npy
uv run scripts/tokenize_dataset.py \
    --vocab scripts/ts_bpe_output/vocab.json \
    --merges scripts/ts_bpe_output/merges.json \
    --input data/TinyStoriesV2-GPT4-valid.txt --output data/tokenized/ts_valid.npy
```

### 1. Train on TinyStories (example, ~17M params)

```sh
uv run scripts/train.py \
    --train-data data/tokenized/ts_train.npy --val-data data/tokenized/ts_valid.npy \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
    --batch-size 64 --max-iters 5000 --lr 3e-4 --warmup-iters 200 \
    --device cuda --dtype bf16 --run-name ts_baseline \
    --checkpoint-path checkpoints/ts_baseline.pt
```

`--dtype` selects the autocast compute dtype (`bf16` default, or `fp16`/`fp32`).
Model params stay fp32, and RMSNorm + the loss are computed in fp32 regardless;
`fp32` disables autocast entirely. Use distinct `--run-name`s to compare dtypes.

Each run is logged to its own timestamped dir,
`experiments/logs/ts_baseline/<timestamp>/{config.json,metrics.jsonl}`, and a
`latest` symlink always points at the most recent run. Per-step metrics
(train/val loss, lr) go to `metrics.jsonl` against gradient steps and wall-clock
time. Plot the latest run with:

```sh
uv run --with matplotlib python scripts/plot_curves.py \
    experiments/logs/ts_baseline/latest --metric val_loss \
    --out experiments/ts_val_loss.png
```

To compare several runs, pass multiple run dirs (e.g.
`experiments/logs/ts_baseline/*/`).

### 2. Train on a larger dataset (OpenWebText)

Use a larger vocab (e.g. 32K) and train a bigger/longer run. Same scripts as
step 0, just pointed at OWT with a larger vocab:

```sh
# Train the BPE tokenizer -> scripts/owt_bpe_output/{vocab.json,merges.json}
uv run python scripts/train_tokenizer.py \
    --input data/owt_train.txt \
    --vocab-size 32000 --out-dir scripts/owt_bpe_output
```

Then tokenize the corpus and train:

```sh
# Tokenize OWT splits -> .npy
uv run scripts/tokenize_dataset.py \
    --vocab scripts/owt_bpe_output/vocab.json \
    --merges scripts/owt_bpe_output/merges.json \
    --input data/owt_train.txt --output data/tokenized/owt_train.npy
uv run scripts/tokenize_dataset.py \
    --vocab scripts/owt_bpe_output/vocab.json \
    --merges scripts/owt_bpe_output/merges.json \
    --input data/owt_valid.txt --output data/tokenized/owt_valid.npy

# Train (longer context, more layers/steps)
uv run scripts/train.py \
    --train-data data/tokenized/owt_train.npy --val-data data/tokenized/owt_valid.npy \
    --vocab-size 32000 --context-length 512 \
    --d-model 768 --num-layers 12 --num-heads 12 --d-ff 2048 \
    --batch-size 32 --max-iters 50000 --lr 3e-4 --warmup-iters 1000 \
    --device cuda --dtype bf16 --run-name owt_baseline \
    --checkpoint-path checkpoints/owt_baseline.pt
```

Plot the curves (same as step 1, pointed at the `owt_baseline` run):

```sh
uv run --with matplotlib python scripts/plot_curves.py \
    experiments/logs/owt_baseline/latest --metric val_loss \
    --out experiments/owt_val_loss.png
```

Tune `--batch-size`, `--max-iters`, and model dims to your GPU memory/budget.
Resume an interrupted run with `--resume checkpoints/<name>.pt`.

### 3. Generate from a trained model

Reconstructs the model from the run's `config.json`, loads the checkpoint and
tokenizer, and samples a completion. Supports temperature scaling and top-p
(nucleus) sampling (use `--temperature 0` for greedy decoding):

```sh
uv run scripts/generate.py \
    --config experiments/logs/ts_baseline/latest/config.json \
    --checkpoint checkpoints/ts_baseline.pt \
    --vocab scripts/ts_bpe_output/vocab.json \
    --merges scripts/ts_bpe_output/merges.json \
    --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.95 --device cuda
```

