# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AI Agent Guidelines for CS336 at Stanford

AI agents should function as teaching aids that help students learn concepts, rather than doing detailed coding for them. Code generation is okay as long as the student understands the concepts.

CS336 is intentionally implementation-heavy. Students are expected to write substantial Python/PyTorch code with limited scaffolding.

### What AI Agents SHOULD Do

* Explain concepts when students are confused by guiding them in the right direction and making sure they build the understanding themselves
* Point students to relevant lecture materials (cs336.stanford.edu), handouts, official documentation, and profiling/debugging tools.
* Review code that students have written and suggest improvements, edge cases, invariants, or debugging checks. Feedback should be general and point students to areas of improvement rather than directly giving solutions.
* Help debug by asking guiding questions rather than providing fixes.
* Explain error messages from Python, PyTorch, CUDA, Triton, and distributed training tools.
* Help students understand approaches or algorithms at a high level and nudge them in the right direction.
* Suggest sanity checks, toy examples, assertions, and profiler-based investigations through active dialog with the student.

### Code Generation Policy

* Yes, AI agents can write code but must ask the student to confirm key concepts and design choices first.
* AI agents should not perform long running implementation sessions to build a full solution — stop in between to verify the student understands the concept.
* Should not complete TODO sections in assignment code without student involvement — stop to verify the student's understanding before moving to the next step.
* Should not commit and push

## Commands

```sh
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_tokenizer.py

# Run a single test by name
uv run pytest tests/test_model.py::test_transformer_lm -s

# Run tests with a timeout (used in make_submission.sh)
uv run pytest --timeout 10 -v ./tests

# Lint
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check

# Run any Python file in the repo (uv activates the venv automatically)
uv run cs336_basics/pretokenization_example.py

# Build submission zip
bash make_submission.sh
```

## Architecture

### The Adapter Pattern

All student implementations live in `cs336_basics/` (the package). The tests never import from `cs336_basics` directly — instead, `tests/adapters.py` is the required bridge: each function in `adapters.py` must be filled in to call the corresponding student implementation. Tests will raise `NotImplementedError` until adapters are wired up.

### Assignment Scope

The assignment builds a GPT-style decoder-only Transformer from scratch. The adapter functions in `tests/adapters.py` define the full scope:

1. **BPE Tokenizer** — `run_train_bpe` (trains vocab/merges from a text file), `get_tokenizer` (constructs a tokenizer from vocab + merges + special tokens)
2. **Neural network primitives** — `run_linear`, `run_embedding`, `run_rmsnorm`, `run_silu`, `run_swiglu`
3. **Attention** — `run_scaled_dot_product_attention`, `run_multihead_self_attention`, `run_rope`, `run_multihead_self_attention_with_rope`
4. **Full model** — `run_transformer_block`, `run_transformer_lm`
5. **Training utilities** — `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`, `get_adamw_cls`, `run_get_lr_cosine_schedule`, `run_get_batch`
6. **Checkpointing** — `run_save_checkpoint`, `run_load_checkpoint`

### Test Infrastructure

Tests use snapshot testing via `tests/conftest.py`:
- `NumpySnapshot` — compares float tensors/arrays against `.npz` files in `tests/_snapshots/` with configurable tolerances (`rtol=1e-4`, `atol=1e-2` by default)
- `Snapshot` — compares arbitrary picklable objects against `.pkl` files

Reference snapshots and model weights are stored in `tests/fixtures/`. The `ts_state_dict` fixture loads a pre-trained TinyStories model for integration tests.

### Data

Training data lives in `data/` (not committed): TinyStories (`TinyStoriesV2-GPT4-{train,valid}.txt`) and OpenWebText sample (`owt_{train,valid}.txt`). See README for download commands.

### Environment

Managed by `uv`. Python 3.12–3.13. Key deps: `torch~=2.11`, `jaxtyping` (tensor shape annotations used throughout adapters), `regex` (for BPE pre-tokenization), `einops`/`einx`.
