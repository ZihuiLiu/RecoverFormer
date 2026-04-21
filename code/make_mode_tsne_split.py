"""
Re-render the latent-mode t-SNE as two stacked vertical panels (2 rows x 1 col)
using pre-computed mode records from results/mode_records_raw.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE

ROOT = Path(r"C:\Users\shiha\Desktop\Research\RecoverFormer")
OUT = ROOT / "figs" / "fig_mode_tsne.pdf"
RAW = ROOT / "results" / "mode_records_raw.json"

FORCE_LEVELS = [50, 100, 150, 200, 250, 300]
FORCE_COLORS = {
    50:  "#4CAF50", 100: "#2196F3", 150: "#FF9800",
    200: "#F44336", 250: "#9C27B0", 300: "#212121",
}


def main():
    records = json.load(open(RAW))
    mean_modes = np.array([r["mean_mode"] for r in records])
    forces = np.array([r["force"] for r in records])
    successes = np.array([r["success"] for r in records])

    perplexity = min(30, len(records) // 4)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    emb = tsne.fit_transform(mean_modes)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.4))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.30)

    # Panel (a): by force
    ax = axes[0]
    for f in FORCE_LEVELS:
        m = forces == f
        ax.scatter(emb[m, 0], emb[m, 1], c=FORCE_COLORS[f], label=f"{f} N",
                   alpha=0.75, s=38, edgecolors="none")
    ax.set_title("(a) t-SNE of per-episode mean mode activations, coloured by push force",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(title="Push force", fontsize=8, title_fontsize=8.5,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Panel (b): by outcome
    ax = axes[1]
    succ = successes == 1
    fail = ~succ
    ax.scatter(emb[succ, 0], emb[succ, 1], c="#1565C0", label=f"Success ({succ.sum()})",
               alpha=0.55, s=36, marker="o", edgecolors="none")
    ax.scatter(emb[fail, 0], emb[fail, 1], c="#B71C1C", label=f"Failure ({fail.sum()})",
               alpha=0.85, s=55, marker="X", edgecolors="none")
    ax.set_title("(b) Same embedding, coloured by recovery outcome",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.savefig(str(OUT), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
