"""Compare ReLU / SiLU (scalar activations) with the SwiGLU gating surface.

Run with:  uv run --with matplotlib python scripts/plot_activations.py
Writes:    scripts/activations_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
# Leave room at the bottom for the explanatory text blocks.
fig.subplots_adjust(bottom=0.34, top=0.88, wspace=0.25)

# --- Left: 1-D scalar activations (directly comparable) ---
x = np.linspace(-6, 6, 400)
ax1.axhline(0, color="gray", lw=0.6)
ax1.axvline(0, color="gray", lw=0.6)
ax1.plot(x, relu(x), label="ReLU(x) = max(0, x)", lw=2)
ax1.plot(x, silu(x), label="SiLU(x) = x·σ(x)", lw=2)
# The scalar "Swish-style gate*value" with identity weights, for intuition only.
ax1.plot(x, silu(x) * x, label="SiLU(x)·x  (gate·value, scalar proxy)",
         lw=2, ls="--")
ax1.set_title("Scalar activations (1-D)")
ax1.set_xlabel("x")
ax1.set_ylabel("output")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.25)

# --- Right: SwiGLU is gating over TWO inputs: SiLU(a) ⊙ b ---
# a = W1·x  (gate path),  b = W3·x  (value path).
a = np.linspace(-6, 6, 300)
b = np.linspace(-6, 6, 300)
A, B = np.meshgrid(a, b)
Z = silu(A) * B  # the elementwise op inside SwiGLU (before W2)

vmax = np.abs(Z).max()
im = ax2.pcolormesh(A, B, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax2.axhline(0, color="k", lw=0.6)
ax2.axvline(0, color="k", lw=0.6)
ax2.set_title("SwiGLU gate:  SiLU(a) · b\n(a = W1·x gate,  b = W3·x value)")
ax2.set_xlabel("a  (gate input)")
ax2.set_ylabel("b  (value input)")
fig.colorbar(im, ax=ax2, label="output")

fig.suptitle("ReLU vs SiLU vs SwiGLU gating", fontsize=15, fontweight="bold")

# --- Explanatory text under each panel (kept with the figure as study notes) ---
left_text = (
    "Scalar activations — apples to apples (functions of one number x):\n"
    "• ReLU: flat 0 for x<0, line x for x>0 — a hard kink at 0.\n"
    "• SiLU = x·σ(x): smooth, dips slightly below 0 near x≈-1\n"
    "  (non-monotonic bump), tracks ReLU for large x. Smoothness\n"
    "  makes it easier to optimize.\n"
    "• SiLU·x (dashed): scalar PROXY for the gate when both paths\n"
    "  share one input. Grows fast (multiplying two signals).\n"
    "  Intuition only — not what SwiGLU literally computes."
)
right_text = (
    "The real SwiGLU op: SiLU(a)·b — a SURFACE over two inputs,\n"
    "not a curve.   a = W1·x (gate),   b = W3·x (value).\n"
    "• Left half (a<0): ~white, output≈0 — gate CLOSED, value b is\n"
    "  suppressed no matter what it is.  ← the on/off gating.\n"
    "• Right half (a>0): output≈b — gate OPEN, the value's sign and\n"
    "  magnitude pass through (red b>0, blue b<0).\n"
    "Takeaway: SiLU is a smoother ReLU; SwiGLU adds a learned,\n"
    "per-channel gate deciding how much of each value to let through."
)
box = dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbbbbb")
fig.text(0.06, 0.02, left_text, fontsize=9.5, va="bottom", ha="left",
         family="monospace", bbox=box)
fig.text(0.55, 0.02, right_text, fontsize=9.5, va="bottom", ha="left",
         family="monospace", bbox=box)

out = "scripts/activations_comparison.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
