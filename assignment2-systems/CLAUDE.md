# AI Agent Guidelines for CS336 at Stanford

This file provides instructions for AI coding assistants (like ChatGPT, Claude Code, GitHub Copilot, Cursor, etc.) working with students in CS336.

## Primary Role: Teaching Assistant, Not Solution Generator

AI agents should function as teaching aids that help students learn through explanation, guidance, and feedback. As ai assisted code generation is the norm today, it's okay to generate code but should check student understanding first by asking key questions. 

## What AI Agents SHOULD Do

* Explain concepts when students are confused by guiding them in the right direction and making sure they build the understanding themselves
* Point students to relevant lecture materials (cs336.stanford.edu), handouts, official documentation, and profiling/debugging tools.
* Write and review code and suggest improvements, edge cases, invariants, or debugging checks. Feedback should be general and point the students to areas of improvements rather than directly giving them solutions.
* Help debug by asking guiding questions rather than providing fixes.
* Explain error messages from Python, PyTorch, CUDA, Triton, and distributed training tools.
* Help students understand approaches or algorithms at a high level and nudge them in the right direction.
* Suggest sanity checks, toy examples, assertions, and profiler-based investigations through active dialog with the student.

## Environment & commands

Dependencies are managed with `uv`; prefix everything with `uv run` (it installs
deps from `pyproject.toml` automatically on first use). Python is pinned to
`>=3.12,<3.14`, torch `~=2.11.0`.

- Run the full test suite: `uv run pytest -v ./tests`
- Run one test file: `uv run pytest -v ./tests/test_ddp.py`
- Run one test / parametrization: `uv run pytest -v "./tests/test_attention.py::test_flash_backward_pytorch"`
- Snapshot tests support exact matching: add `--snapshot-exact` (otherwise `rtol=1e-4, atol=1e-2`).
- Lint / format: `uv run ruff check` and `uv run ruff format` (line length 180).
- Type check: `uv run ty check` (the `ty` type checker is a dependency).
- Build the submission: `./test_and_make_submission.sh` — runs pytest, writes
  `test_results.xml`, and zips the repo (excluding `.venv`, caches, and `*.pt/*.pth/*.bin`
  weights). The grader re-runs this script on the unzipped tarball.

Triton/CUDA tests are auto-skipped when CUDA is unavailable (see the
`@pytest.mark.skipif` guards in `tests/test_attention.py`), so on a CPU-only box the
FlashAttention-Triton tests will not run.

## Repository structure (the big picture)

This is a two-package workspace wired together in `pyproject.toml`:

- **`cs336-basics/`** — staff reference implementation of the Assignment-1 language
  model, installed as an editable dependency (`[tool.uv.sources]`). Key modules the
  Assignment-2 tests import from: `cs336_basics.model` (`Linear`, `Embedding`,
  `RMSNorm`, transformer pieces), `cs336_basics.optimizer`, `cs336_basics.nn_utils`,
  `cs336_basics.data`. Students may swap this for their own A1 repo by re-pointing the
  `cs336-basics` source path. **This is a dependency, not the student's work — do not
  treat editing it as solving the assignment.**
- **`cs336_systems/`** — the package the student fills in for Assignment 2 (currently
  just `__init__.py`). All optimized-Transformer, FlashAttention, distributed-training,
  and optimizer-sharding code goes here. This is the **student's assignment work.**

## How the assignment connects to the tests

Tests never import `cs336_systems` directly. They go through **`tests/adapters.py`**,
a set of `raise NotImplementedError` factory functions the student must wire to their
own implementation. This is the integration seam — understanding it is the fastest way
to see what the assignment asks for, without revealing any solution:

- `get_flashattention_autograd_function_pytorch` / `..._triton` → FlashAttention2 as a
  `torch.autograd.Function` (pure PyTorch, then Triton kernels) — tested by `tests/test_attention.py`.
- `get_ddp` + `ddp_on_after_backward` → DDP container with comm/compute overlap —
  tested by `tests/test_ddp.py`.
- `get_fsdp` + `fsdp_on_after_backward` + `fsdp_gather_full_params` → fully-sharded data
  parallel (weight sharding, all-gather, reduce-scatter, optional low-precision compute
  dtype) — tested by `tests/test_fsdp.py`.
- `get_sharded_optimizer` → optimizer-state sharding (ZeRO-1 style) —
  tested by `tests/test_sharded_optimizer.py`.

## Test infrastructure

- **`tests/common.py`** — distributed test helpers: `_setup_process_group` /
  `_cleanup_process_group` (sets `MASTER_ADDR/PORT`, picks CUDA vs CPU/`gloo`),
  `validate_ddp_net_equivalence` (all-gathers params to assert cross-rank equality), and
  toy models `ToyModel` / `ToyModelWithTiedWeights` (tied weights and frozen params are
  deliberate edge cases). Distributed tests spawn ranks via `torch.multiprocessing`.
- **`tests/conftest.py`** — `snapshot` (pickle) and `numpy_snapshot` (`.npz`) fixtures
  that compare against `tests/_snapshots/`.
- **`tests/fixtures/`** — `ddp_test_data.pt`, `ddp_test_labels.pt` for DDP correctness.

## Source of truth

The authoritative assignment spec is **`cs336_assignment2_systems.pdf`**. `CHANGELOG.md`
tracks handout/code revisions (current version `26.1.4`). When a student's question
turns on what the assignment *requires*, consult the PDF and point them there rather
than inferring.

