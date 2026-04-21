"""
Vertical (portrait) hero teaser: single-column MuJoCo render of the Unitree G1
mid-push, with a force arrow and compact annotations. Sized to fit in a single
IEEE-template column on the front page.
"""
import sys, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import deque

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs.g1_recovery_env import G1RecoveryEnv, OBS_DIM_BASE, N_ACTUATORS
from models.recoverformer import RecoverFormer
from train import RunningMeanStd

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "code" / "logs" / "recoverformer_v4" / "recoverformer_open_floor_20260415_195523" / "final.pt"
OUT = ROOT / "figs" / "fig_teaser.pdf"

# Portrait render: taller than wide (framebuffer max height is 480)
RENDER_W = 280
RENDER_H = 480


def load_model(device):
    ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
    args = ckpt["args"]
    model = RecoverFormer(
        obs_dim=OBS_DIM_BASE, action_dim=N_ACTUATORS,
        embed_dim=args.get("embed_dim", 256),
        n_heads=args.get("n_heads", 4),
        n_layers=args.get("n_layers", 4),
        n_modes=args.get("n_modes", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    obs_rms = RunningMeanStd(shape=(OBS_DIM_BASE,))
    if "obs_rms_mean" in ckpt:
        obs_rms.mean = ckpt["obs_rms_mean"]
        obs_rms.var = ckpt["obs_rms_var"]
        obs_rms.count = ckpt["obs_rms_count"]
    return model, obs_rms, args.get("history_len", 50)


def render_hero(seed=11, push_force=180.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, obs_rms, history_len = load_model(device)

    np.random.seed(seed); torch.manual_seed(seed)

    env = G1RecoveryEnv(
        env_type="open_floor",
        max_episode_steps=400,
        render_mode="rgb_array",
        push_force_range=(0, 0),
        push_interval_range=(9999, 9999),
    )
    obs, _ = env.reset(seed=seed)
    history = deque(maxlen=history_len)

    env._renderer.close()
    env._renderer = mujoco.Renderer(env.model, height=RENDER_H, width=RENDER_W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance = 2.35
    cam.azimuth = 75.0
    cam.elevation = -4.0
    cam.lookat[:] = [0.15, 0.0, 0.75]

    PUSH_STEP = 40
    PUSH_DIR = np.array([1.0, 0.25, 0.0])
    PUSH_DIR = PUSH_DIR / np.linalg.norm(PUSH_DIR)
    PUSH_DURATION = 5
    HERO_STEP = PUSH_STEP + 8

    hero_frame = None
    for step in range(200):
        norm = obs_rms.normalize(obs)
        history.append(norm)
        hl = list(history)
        while len(hl) < history_len:
            hl.insert(0, hl[0])
        x = torch.tensor(np.stack(hl)[np.newaxis], dtype=torch.float32, device=device)
        with torch.no_grad():
            a = model(x)["action"].cpu().numpy()[0]

        if PUSH_STEP <= step < PUSH_STEP + PUSH_DURATION:
            env.data.xfrc_applied[env.model.body("torso_link").id, :3] = push_force * PUSH_DIR
        else:
            env.data.xfrc_applied[env.model.body("torso_link").id, :3] = 0.0

        obs, _, term, _, _ = env.step(a)

        if step == HERO_STEP:
            env._renderer.update_scene(env.data, camera=cam)
            hero_frame = env._renderer.render()
            break
        if term:
            break

    env.close()
    return hero_frame


def composite(img):
    # Portrait figure: fits one IEEE column (~3.4 inch wide), ~4.9 tall
    fig = plt.figure(figsize=(3.4, 5.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    h, w = img.shape[:2]

    # Title banner across the top
    ax.text(w * 0.5, h * 0.045,
            "RecoverFormer",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="#333", linewidth=1.0, alpha=0.92),
            zorder=6)

    # Push arrow: horizontal into torso (torso is roughly at ~0.35-0.50 vertical)
    arrow = FancyArrowPatch(
        (w * 0.04, h * 0.40), (w * 0.32, h * 0.40),
        arrowstyle="-|>,head_width=10,head_length=13",
        color="#D32F2F", linewidth=5.0, mutation_scale=1.0,
        zorder=5,
    )
    ax.add_patch(arrow)
    ax.text(w * 0.02, h * 0.345, "F", fontsize=22, fontweight="bold",
            color="#D32F2F", zorder=6)
    ax.text(w * 0.17, h * 0.455, "180 N",
            fontsize=9, color="#D32F2F", fontweight="bold", zorder=6, ha="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#D32F2F", linewidth=1.0, alpha=0.92))

    # Recovery annotation on the right side
    ax.annotate(
        "balance\nrecovery",
        xy=(w * 0.62, h * 0.36), xytext=(w * 0.80, h * 0.15),
        fontsize=9, color="#1565C0", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.3,
                        connectionstyle="arc3,rad=0.20"),
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                  edgecolor="#1565C0", linewidth=1.0, alpha=0.92),
        zorder=6,
    )

    # Bottom caption
    ax.text(w * 0.5, h * 0.965,
            "end-to-end transformer policy, 50 Hz",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="#888", linewidth=0.8, alpha=0.88),
            zorder=6)

    plt.savefig(str(OUT), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=200,
                bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    for seed in [11, 3, 7, 17, 23, 29, 42, 101, 202, 303, 404, 505, 606, 707]:
        img = render_hero(seed=seed)
        if img is not None:
            print(f"Captured hero at seed={seed}")
            composite(img)
            break
    else:
        print("No seed produced a hero frame; aborting.")
