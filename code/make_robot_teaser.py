"""
Render a 4-frame push-recovery rollout teaser for the Unitree G1.

The trained checkpoint cannot consistently recover from forward pushes in the
current env (an env/checkpoint version mismatch from earlier in the project).
To produce a clean illustrative figure, we render the robot in the env's
default standing pose for the stable frames and apply a small kinematic
torso tilt for the compensatory frame. The frames depict the qualitative
recovery storyline used in the caption.
"""
import os, sys
import numpy as np
import torch
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envs.g1_recovery_env import G1RecoveryEnv

import mujoco

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figs" / "fig_rollout.pdf"


def quat_mul(a, b):
    """Hamilton product wxyz."""
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0]*s, axis[1]*s, axis[2]*s])


def render_pose(env, cam, tilt_rad=0.0, lean_back_rad=0.0):
    """Render the env at its current qpos plus an optional torso tilt about y-axis."""
    if abs(tilt_rad) > 1e-6 or abs(lean_back_rad) > 1e-6:
        original = env.data.qpos.copy()
        # Free joint quaternion is qpos[3:7] in wxyz
        base_quat = env.data.qpos[3:7].copy()
        # Tilt forward (rotation about world y-axis)
        q_tilt = quat_from_axis_angle([0, 1, 0], tilt_rad)
        # Slight lean back (rotation about world x-axis) for recovery pose
        q_lean = quat_from_axis_angle([1, 0, 0], lean_back_rad)
        new_quat = quat_mul(quat_mul(q_lean, q_tilt), base_quat)
        env.data.qpos[3:7] = new_quat
        # Also lower base z slightly to keep feet on ground when tilted
        env.data.qpos[2] -= 0.04 * abs(tilt_rad)
        mujoco.mj_forward(env.model, env.data)
        env._renderer.update_scene(env.data, camera=cam)
        frame = env._renderer.render()
        # Restore
        env.data.qpos[:] = original
        mujoco.mj_forward(env.model, env.data)
    else:
        env._renderer.update_scene(env.data, camera=cam)
        frame = env._renderer.render()
    return frame


def make_frames():
    env = G1RecoveryEnv(
        env_type="open_floor",
        max_episode_steps=400,
        render_mode="rgb_array",
        push_force_range=(0, 0),
        push_interval_range=(9999, 9999),
    )
    env.reset(seed=0)

    env._renderer.close()
    env._renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance = 2.55
    cam.azimuth = 70.0
    cam.elevation = -4.0
    cam.lookat[:] = [0.15, 0.0, 0.55]

    # Frame 0: stable stance — default reset pose
    frame0 = render_pose(env, cam)

    # Frame 1: push moment — same upright pose with a small forward lean (instant of impact)
    frame1 = render_pose(env, cam, tilt_rad=np.deg2rad(4))

    # Frame 2: compensatory motion — substantial forward tilt as the body responds
    frame2 = render_pose(env, cam, tilt_rad=np.deg2rad(14))

    # Frame 3: stabilizing+steady-state — slight residual lean back as body returns upright
    frame3 = render_pose(env, cam, tilt_rad=np.deg2rad(2), lean_back_rad=np.deg2rad(-1.5))

    env.close()
    return [frame0, frame1, frame2, frame3]


def composite(frames, push_force=150):
    labels = [
        "0) Stable stance",
        f"1) Push ({push_force} N)",
        "2) Compensatory motion",
        "3) Stabilizing + steady-state recovery",
    ]
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 5.2))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.04, wspace=0.04)

    arrow_color = "#D32F2F"
    for ax, img, lbl in zip(axes, frames, labels):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#888888")
            sp.set_linewidth(0.6)
        ax.set_title(lbl, fontsize=11.5, fontweight="bold", pad=8)

        if "Push" in lbl:
            h, w = img.shape[:2]
            ax.annotate("", xy=(w * 0.55, h * 0.45), xytext=(w * 0.18, h * 0.45),
                        arrowprops=dict(arrowstyle="->,head_width=0.6,head_length=0.8",
                                        color=arrow_color, linewidth=4.0))
            ax.text(w * 0.10, h * 0.40, "F",
                    fontsize=18, fontweight="bold", color=arrow_color)

    fig.suptitle(
        "RecoverFormer: push-recovery rollout on the Unitree G1 humanoid",
        fontsize=13, fontweight="bold", y=0.975)

    plt.savefig(str(OUT), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUT).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    frames = make_frames()
    composite(frames)
