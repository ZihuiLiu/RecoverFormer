"""
Modern AI-style architecture diagram for RecoverFormer:
input tokenization -> positional embedding -> causal transformer blocks
-> latent mode head + contact affordance head -> action decoder.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(r"C:\Users\shiha\Desktop\Research\RecoverFormer\figs\fig_architecture.pdf")

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG_IN      = "#E3F2FD"
C_TOKEN      = "#4A6CFF"
C_POSENC     = "#9C27B0"
C_ADD        = "#F5F5F5"
C_BLOCK      = "#E8EAF6"
C_ATTN       = "#3949AB"
C_FFN        = "#1565C0"
C_NORM       = "#90A4AE"
C_MODE       = "#2E7D32"
C_AFF        = "#EF6C00"
C_ACTION     = "#6A1B9A"
C_OUT        = "#FAFAFA"
C_ARROW      = "#333333"
C_ARROW_LITE = "#888888"


def rbox(ax, cx, cy, w, h, color, label, sublabel="", fontsize=9,
         textcolor="white", edgecolor="none", alpha=1.0, lw=0.0):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=color, edgecolor=edgecolor, linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(cx, cy + h * 0.20, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
        ax.text(cx, cy - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color=textcolor, alpha=0.85, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)


def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.3, style="-|>", hw=0.4, hl=0.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"{style},head_width={hw},head_length={hl}",
                                color=color, lw=lw), zorder=2)


fig, ax = plt.subplots(figsize=(7.6, 9.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11.4)
ax.axis("off")

# ── Row 0 · 50-step observation history ───────────────────────────────────────
hist_y = 0.75
ax.text(5.0, hist_y + 0.75,
        r"Observation history  $\mathcal{H}_t = \{o_{t-H+1}, \ldots, o_{t-1}, o_t\}$  "
        r"($H = 50$, $\dim(o) = 106$)",
        ha="center", va="center", fontsize=12, color="#1a1a1a", fontweight="bold")

n_tokens = 8       # visual stand-in for 50 frames
token_w = 0.62
token_h = 0.50
gap = 0.18
start_x = 5.0 - (n_tokens * token_w + (n_tokens - 1) * gap) / 2 + token_w / 2
for i in range(n_tokens):
    cx = start_x + i * (token_w + gap)
    alpha = 0.35 + 0.65 * (i / (n_tokens - 1))
    rbox(ax, cx, hist_y, token_w, token_h, C_BG_IN, "",
         edgecolor="#1565C0", alpha=alpha, lw=0.8)
    # mini-sparkline to suggest observation content
    xs = [cx - token_w * 0.35 + j * token_w * 0.1 for j in range(8)]
    ys = [hist_y + 0.04 * ((j % 3) - 1) for j in range(8)]
    ax.plot(xs, ys, color="#1565C0", lw=0.7, alpha=0.6)
# ellipsis between frames
ax.text(start_x - token_w * 0.8, hist_y, r"$\cdots$",
        ha="center", va="center", fontsize=14, color="#1565C0")
ax.text(start_x + (n_tokens - 1) * (token_w + gap) + token_w * 0.8, hist_y,
        r"$t$", ha="center", va="center", fontsize=12, color="#1565C0")
ax.text(start_x - token_w * 0.8 - 0.45, hist_y,
        r"$t{-}H{+}1$", ha="center", va="center", fontsize=12, color="#1565C0")

# ── Row 1 · Linear tokenization + positional encoding ─────────────────────────
tok_y = 2.7
rbox(ax, 3.2, tok_y, 2.8, 0.75, C_TOKEN,
     "Linear Tokenizer",
     r"$x_i = W_e o_i + b_e \in \mathbb{R}^d$",
     fontsize=11.5)
rbox(ax, 6.8, tok_y, 2.8, 0.75, C_POSENC,
     "Positional Encoding",
     r"$p_i \in \mathbb{R}^d$  (sinusoidal)",
     fontsize=11.5)
# sum node
sum_xy = (5.0, 4.1)
circ = plt.Circle(sum_xy, 0.22, facecolor=C_ADD, edgecolor="#333", lw=1.0, zorder=3)
ax.add_patch(circ)
ax.text(*sum_xy, "+", ha="center", va="center", fontsize=15, fontweight="bold", zorder=4)
arr(ax, 3.2, tok_y + 0.38, 4.80, 4.1 - 0.20, color=C_ARROW_LITE, lw=1.0)
arr(ax, 6.8, tok_y + 0.38, 5.20, 4.1 - 0.20, color=C_ARROW_LITE, lw=1.0)
ax.text(5.0, 4.50, r"$h_i^{(0)} = x_i + p_i$",
        ha="center", va="bottom", fontsize=11, style="italic", color="#333")

# arrows from history tokens up to tokenizer / posenc
arr(ax, 5.0, hist_y + token_h / 2 + 0.02, 5.0, tok_y - 0.45, color=C_ARROW, lw=1.3)

# ── Row 2 · Causal transformer stack ──────────────────────────────────────────
stack_cy = 6.55
stack_w, stack_h = 6.4, 3.0
rbox(ax, 5.0, stack_cy, stack_w, stack_h, C_BLOCK,
     "", edgecolor="#1a237e", alpha=1.0, lw=1.2)
ax.text(5.0, stack_cy + stack_h / 2 - 0.28,
        r"Causal Transformer Encoder  ($L{=}4$, $d{=}256$, heads${=}4$)",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#1a237e")

# sub-components inside the stack
inner_cy = stack_cy - 0.15
# LayerNorm (pre)
rbox(ax, 2.55, inner_cy, 1.3, 0.55, C_NORM, "LayerNorm",
     fontsize=10.5, textcolor="white")
# Multi-head Causal Attention
rbox(ax, 4.35, inner_cy, 2.05, 0.85, C_ATTN,
     "Masked Multi-Head\nSelf-Attention",
     fontsize=10.5, textcolor="white")
# FFN
rbox(ax, 6.60, inner_cy, 1.70, 0.85, C_FFN,
     "FFN\n(GELU, $4d$)",
     fontsize=10.5, textcolor="white")
# Residual indicator (arrow skipping from left to right)
arr(ax, 3.22, inner_cy, 3.33, inner_cy, color=C_ARROW, lw=1.3)
arr(ax, 5.40, inner_cy, 5.76, inner_cy, color=C_ARROW, lw=1.3)
# residual loops
ax.annotate("", xy=(4.35, inner_cy + 0.95), xytext=(2.55, inner_cy + 0.95),
            arrowprops=dict(arrowstyle="-", color="#777", lw=0.9, linestyle="dotted"))
ax.annotate("", xy=(4.35, inner_cy + 0.95), xytext=(4.35, inner_cy + 0.45),
            arrowprops=dict(arrowstyle="->", color="#777", lw=0.9, linestyle="dotted"))
ax.text(3.45, inner_cy + 1.08, "residual", fontsize=9, style="italic", color="#666")

# xL indicator on the right
ax.text(8.00, stack_cy, r"$\times L$", fontsize=16, fontweight="bold",
        ha="left", va="center", color="#1a237e")

# Attention equation underneath
ax.text(4.35, inner_cy - 0.78,
        r"$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V$",
        ha="center", va="center", fontsize=10.5, style="italic", color="#1a1a1a")

# arrow from sum into stack
arr(ax, 5.0, 4.33, 5.0, stack_cy - stack_h / 2 - 0.02, color=C_ARROW, lw=1.4)

# ── Encoder output label ──────────────────────────────────────────────────────
enc_out_y = stack_cy + stack_h / 2 + 0.35
ax.text(5.0, enc_out_y, r"$e_t = h_t^{(L)} \in \mathbb{R}^{256}$",
        ha="center", va="center", fontsize=11.5, color="#333", style="italic")

# ── Row 3 · Three branches: mode head | decoder | affordance head ────────────
heads_cy = 9.55
rbox(ax, 1.85, heads_cy, 3.0, 0.90, C_MODE,
     "Latent Mode Head",
     r"$p_\theta(z_t|e_t)=\mathrm{softmax}(W_z e_t)$, $K{=}4$",
     fontsize=10.5)

rbox(ax, 5.0, heads_cy, 2.3, 0.90, C_ACTION,
     "Action Decoder",
     r"MLP$([e_t;\mathrm{emb}(z_t);c_t])$",
     fontsize=10.5)

rbox(ax, 8.15, heads_cy, 3.0, 0.90, C_AFF,
     "Contact Affordance Head",
     r"$c_t = \sigma(W_c e_t)\in[0,1]^{K_c}$, $K_c{=}8$",
     fontsize=10.5)

# branching arrows from encoder out to the three heads
arr(ax, 5.0, enc_out_y + 0.20, 1.85, heads_cy - 0.50, color=C_MODE, lw=1.3)
arr(ax, 5.0, enc_out_y + 0.20, 5.0, heads_cy - 0.50, color=C_ARROW, lw=1.3)
arr(ax, 5.0, enc_out_y + 0.20, 8.15, heads_cy - 0.50, color=C_AFF, lw=1.3)

# arrows from mode / affordance into decoder
arr(ax, 3.37, heads_cy, 3.86, heads_cy, color=C_MODE, lw=1.2)
arr(ax, 6.65, heads_cy, 6.17, heads_cy, color=C_AFF, lw=1.2)

# ── Row 4 · Output action ────────────────────────────────────────────────────
action_cy = 10.95
rbox(ax, 5.0, action_cy, 2.5, 0.55, C_OUT,
     r"Action $a_t \in \mathbb{R}^{29}$  (joint targets, 50 Hz)",
     fontsize=11.5, textcolor="#1a1a1a", edgecolor="#6A1B9A", lw=1.0)
arr(ax, 5.0, heads_cy + 0.48, 5.0, action_cy - 0.30, color=C_ARROW, lw=1.3)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=C_TOKEN,  label="Tokenizer"),
    mpatches.Patch(facecolor=C_POSENC, label="Positional Encoding"),
    mpatches.Patch(facecolor=C_ATTN,   label="Self-Attention"),
    mpatches.Patch(facecolor=C_FFN,    label="Feed-Forward"),
    mpatches.Patch(facecolor=C_MODE,   label="Latent Mode"),
    mpatches.Patch(facecolor=C_AFF,    label="Affordance"),
    mpatches.Patch(facecolor=C_ACTION, label="Action Decoder"),
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.03), fontsize=9.5, ncol=4,
          framealpha=0.0, edgecolor="none",
          handlelength=1.3, handletextpad=0.5, columnspacing=1.2)

plt.tight_layout(pad=0.25)
plt.savefig(str(OUT), dpi=300, bbox_inches="tight")
plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")
