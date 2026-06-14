"""Visualize RoPE: position becomes rotation angle, and q·k depends only on (m-n).

Run with:  uv run --with matplotlib python scripts/plot_rope.py
Writes:    scripts/rope_relative_position.png
"""

import numpy as np
import matplotlib.pyplot as plt


def rot(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, -s], [s, c]])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
fig.subplots_adjust(bottom=0.42, top=0.90, wspace=0.28)

# --- Left: one 2-D vector rotated by angle = position * theta (clock hands) ---
theta = np.deg2rad(35)          # per-position rotation for this pair
q = np.array([1.0, 0.0])        # the base 2-D sub-vector
positions = range(6)
cmap = plt.cm.viridis
ax1.axhline(0, color="gray", lw=0.6)
ax1.axvline(0, color="gray", lw=0.6)
for p in positions:
    v = rot(p * theta) @ q
    color = cmap(p / 5)
    ax1.annotate("", xy=v, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=color, lw=2.2))
    ax1.text(v[0] * 1.12, v[1] * 1.12, f"p={p}", color=color,
             fontsize=10, ha="center", va="center")
circle = plt.Circle((0, 0), 1.0, fill=False, ls="--", color="lightgray")
ax1.add_patch(circle)
ax1.set_xlim(-1.3, 1.3)
ax1.set_ylim(-1.3, 1.3)
ax1.set_aspect("equal")
ax1.set_title("Position → rotation angle\n(same vector, rotated by p·θ)")
ax1.set_xlabel("dim 2i")
ax1.set_ylabel("dim 2i+1")

# --- Right: dot product <R(m*theta)q, R(n*theta)k> over (m, n) grid ---
# Diagonal stripes prove it depends only on (n - m), not absolute m or n.
k = np.array([0.8, 0.5])
N = 16
M = np.zeros((N, N))
for m in range(N):
    for n in range(N):
        qm = rot(m * theta) @ q
        kn = rot(n * theta) @ k
        M[n, m] = qm @ kn          # row=n (key pos), col=m (query pos)

im = ax2.imshow(M, origin="lower", cmap="RdBu_r",
                vmin=-np.abs(M).max(), vmax=np.abs(M).max())
ax2.set_title("Attention score  ⟨R(mθ)q, R(nθ)k⟩\nconstant along each diagonal (n−m = const)")
ax2.set_xlabel("m  (query position)")
ax2.set_ylabel("n  (key position)")
fig.colorbar(im, ax=ax2, label="dot product")

fig.suptitle("RoPE: absolute positions cancel, only relative position (n−m) survives",
             fontsize=14, fontweight="bold")

left_text = (
    "Each 2-D pair of a vector is ROTATED, not shifted.\n"
    "• Angle of rotation = position p × θ  (θ = pair frequency).\n"
    "• Position becomes an angle, like a clock hand sweeping\n"
    "  around the circle as p grows.\n"
    "• Length is preserved — RoPE never changes a vector's norm,\n"
    "  it only turns it. Here θ=35° per step for illustration."
)
right_text = (
    "Why it gives RELATIVE position:\n"
    "⟨R(mθ)q, R(nθ)k⟩ = qᵀR(mθ)ᵀR(nθ)k = qᵀR((n−m)θ)k\n"
    "because R(α)ᵀ=R(−α) and R(a)R(b)=R(a+b).\n"
    "• The absolute m, n cancel; only (n−m) remains.\n"
    "• In the heatmap the color is constant along each diagonal\n"
    "  (fixed n−m) — same score regardless of where the pair sits."
)
box = dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#bbbbbb")
fig.text(0.06, 0.14, left_text, fontsize=9.5, va="bottom", ha="left",
         family="monospace", bbox=box)
fig.text(0.55, 0.14, right_text, fontsize=9.5, va="bottom", ha="left",
         family="monospace", bbox=box)

example_text = (
    'Example — "he passes the pass": the token "pass" appears twice with identical Q/K vectors before position info.\n'
    "RoPE rotates each by a different position-dependent angle, so the two copies point in different directions and become\n"
    "distinguishable by relative order. RoPE only supplies the positional signal — the model LEARNS to use it to tell the\n"
    "verb \"pass\" from the noun \"pass\"."
)
ex_box = dict(boxstyle="round", facecolor="#fff7e6", edgecolor="#e0a800")
fig.text(0.5, 0.02, example_text, fontsize=9.5, va="bottom", ha="center",
         family="monospace", bbox=ex_box)

out = "scripts/rope_relative_position.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
