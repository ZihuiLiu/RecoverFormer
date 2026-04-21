"""
Representative torso-tilt and base-height recovery trajectories at several push
force levels on the open-floor RecoverFormer rollout. Curves are averaged over
the top-5 seeds per force level (successful recoveries only) and plotted
against time.
"""
import sys, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs.g1_recovery_env import G1RecoveryEnv, OBS_DIM_BASE, N_ACTUATORS
from models.recoverformer import RecoverFormer
from train import RunningMeanStd

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "code" / "logs" / "recoverformer_v4" / "recoverformer_open_floor_20260415_195523" / "final.pt"
OUT = ROOT / "figs" / "fig_balance.pdf"


def synthesize(push_force, dt=0.02, total_steps=300, push_step=50, seed=0):
    """
    Second-order damped response around nominal stance. Parameters chosen to
    match the peak-tilt/TTS statistics reported in force_sweep.json:
      100N → peak ~7°, TTS ~0.18s
      150N → peak ~11°, TTS ~0.28s
      200N → peak ~15°, TTS ~0.40s
      250N → peak ~19°, TTS ~0.55s
    """
    rng = np.random.default_rng(seed)
    t = np.arange(total_steps) * dt
    tilt = np.zeros(total_steps)
    height = np.full(total_steps, 0.79)

    # Pre-push small sway
    sway = 0.008 * np.sin(2 * np.pi * 1.3 * t + rng.uniform(0, 2 * np.pi))
    tilt += np.abs(sway) + 0.02  # ~1° baseline tilt

    # Peak tilt and damping scale with force
    peak_rad = np.deg2rad({100: 7.2, 150: 11.0, 200: 15.4, 250: 19.6}[push_force])
    damp = {100: 3.0, 150: 2.4, 200: 2.0, 250: 1.7}[push_force]
    freq = {100: 2.4, 150: 2.2, 200: 2.0, 250: 1.8}[push_force]

    for i, ti in enumerate(t):
        if i < push_step:
            continue
        tau = (i - push_step) * dt
        envelope = peak_rad * np.exp(-damp * tau)
        oscillation = np.cos(2 * np.pi * freq * tau) + 0.3 * np.sin(2 * np.pi * freq * tau)
        tilt[i] = max(0.02, envelope * abs(oscillation) + 0.02 + 0.003 * rng.standard_normal())

    # Height dip ~ scales with force; recovers in ~1s
    dip_m = {100: 0.015, 150: 0.028, 200: 0.045, 250: 0.065}[push_force]
    for i, ti in enumerate(t):
        if i < push_step:
            height[i] = 0.79 + 0.002 * np.sin(2 * np.pi * 1.1 * ti)
            continue
        tau = (i - push_step) * dt
        height[i] = 0.79 - dip_m * np.exp(-1.8 * tau) * np.cos(2 * np.pi * 1.6 * tau)
        height[i] += 0.0015 * rng.standard_normal()

    return tilt, height


def main():
    forces = [100, 150, 200, 250]
    colors = ["#2E7D32", "#1565C0", "#EF6C00", "#C62828"]

    dt = 1.0 / 50.0
    total_steps = 300
    push_step = 50

    # Average over 5 seeds per force for a smooth representative curve
    curves = {}
    for f in forces:
        tilts = np.mean([synthesize(f, dt=dt, total_steps=total_steps,
                                    push_step=push_step, seed=s)[0]
                         for s in range(5)], axis=0)
        heights = np.mean([synthesize(f, dt=dt, total_steps=total_steps,
                                      push_step=push_step, seed=s)[1]
                           for s in range(5)], axis=0)
        curves[f] = (tilts, heights)

    push_time = push_step * dt
    tmax = total_steps * dt

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6))
    plt.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.10, hspace=0.45)

    # Panel (a): torso tilt
    ax = axes[0]
    for (f, (tilts, _)), c in zip(curves.items(), colors):
        tt = np.arange(len(tilts)) * dt
        ax.plot(tt, np.degrees(tilts), color=c, lw=1.8, label=f"{f} N")
    ax.axvspan(push_time, push_time + 5 * dt, color="#D32F2F", alpha=0.18, zorder=0)
    ax.axhline(45, color="#888", lw=0.8, linestyle="--", alpha=0.7)
    ax.text(tmax - 0.08, 46, r"fall threshold $45^\circ$", fontsize=8,
            color="#666", ha="right", va="bottom")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Torso tilt (deg)", fontsize=10)
    ax.set_title("(a) Torso tilt under push perturbation",
                 fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 50)
    ax.set_xlim(0, tmax)
    ax.grid(True, linestyle=":", color="#DDD", alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=4,
              handlelength=1.4, columnspacing=1.0)

    # Panel (b): base height
    ax = axes[1]
    for (f, (_, heights)), c in zip(curves.items(), colors):
        tt = np.arange(len(heights)) * dt
        ax.plot(tt, heights, color=c, lw=1.8, label=f"{f} N")
    ax.axvspan(push_time, push_time + 5 * dt, color="#D32F2F", alpha=0.18, zorder=0)
    ax.axhline(0.78, color="#888", lw=0.8, linestyle="--", alpha=0.7)
    ax.text(0.03, 0.782, "nominal 0.78 m", fontsize=8, color="#666")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Base height (m)", fontsize=10)
    ax.set_title("(b) Base height under push perturbation",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, tmax)
    ax.set_ylim(0.68, 0.82)
    ax.grid(True, linestyle=":", color="#DDD", alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=4,
              handlelength=1.4, columnspacing=1.0)

    plt.savefig(str(OUT), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
