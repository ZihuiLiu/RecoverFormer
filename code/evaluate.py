"""
Comprehensive evaluation script for RecoverFormer paper experiments.

Produces:
  - Table 1: RSR / TTS / PTT per push force magnitude (open floor)
  - Table 2: RSR / CER per environment type (walled / cluttered)
  - Table 3: RSR under domain mismatch (friction, mass, latency)
  - Sequential push RSR (double-push mid-recovery stress test)

Usage:
    # Table 1 — force sweep
    python evaluate.py --recoverformer ../logs/recoverformer_full/*/final.pt \
                       --baseline      ../logs/baseline_flat/*/final.pt \
                       --experiment force_sweep

    # Sequential push stress test
    python evaluate.py --recoverformer ../logs/recoverformer_full/*/final.pt \
                       --baseline      ../logs/baseline_flat/*/final.pt \
                       --experiment sequential_push

    # Domain mismatch robustness
    python evaluate.py --recoverformer ../logs/recoverformer_full/*/final.pt \
                       --baseline      ../logs/baseline_flat/*/final.pt \
                       --experiment domain_mismatch

    # Contact-aware (walled environment)
    python evaluate.py --recoverformer ../logs/recoverformer_walled/*/final.pt \
                       --baseline      ../logs/baseline_walled/*/final.pt \
                       --experiment contact_aware --env_type walled
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent))

from envs.g1_recovery_env import G1RecoveryEnv, OBS_DIM_BASE, N_ACTUATORS
from models.recoverformer import RecoverFormer
from models.baseline_mlp import BaselineMLP
from train import RunningMeanStd

# ── Constants ─────────────────────────────────────────────────────────────────

VELOCITY_STABLE_THRESHOLD = 0.15   # m/s — below this = "stabilized"
TILT_FAIL_THRESHOLD_DEG   = 45.0   # degrees — above this = "fell"
MAX_EVAL_STEPS            = 500    # 10 seconds at 50 Hz


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    if args["model"] == "recoverformer":
        model = RecoverFormer(
            obs_dim=OBS_DIM_BASE,
            action_dim=N_ACTUATORS,
            embed_dim=args.get("embed_dim", 256),
            n_heads=args.get("n_heads", 4),
            n_layers=args.get("n_layers", 4),
            n_modes=args.get("n_modes", 4),
        ).to(device)
        use_history = True
        history_len = args.get("history_len", 50)
    else:
        model = BaselineMLP(obs_dim=OBS_DIM_BASE, action_dim=N_ACTUATORS).to(device)
        use_history = False
        history_len = 1

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    obs_rms = RunningMeanStd(shape=(OBS_DIM_BASE,))
    if "obs_rms_mean" in ckpt:
        obs_rms.mean = ckpt["obs_rms_mean"]
        obs_rms.var  = ckpt["obs_rms_var"]
        obs_rms.count = ckpt["obs_rms_count"]

    return model, args["model"], obs_rms, use_history, history_len


def get_action(model, obs_rms, history, obs, use_history, history_len, device,
               tta_optimizer=None, prev_encoding=None, prev_action=None):
    """
    Get action with optional test-time adaptation.

    If tta_optimizer is provided (RecoverFormer only):
      1. Compute adaptation loss using prev_encoding + prev_action → predicted next encoding
      2. Update only adaptation module parameters
      3. Apply adaptation residual to action
    """
    norm_obs = obs_rms.normalize(obs)
    if use_history:
        history.append(norm_obs)
        hist_list = list(history)
        while len(hist_list) < history_len:
            hist_list.insert(0, hist_list[0])
        obs_t = torch.tensor(np.stack(hist_list)[np.newaxis], dtype=torch.float32, device=device)
    else:
        obs_t = torch.tensor(norm_obs[np.newaxis], dtype=torch.float32, device=device)

    # TTA update step (RecoverFormer only)
    if tta_optimizer is not None and prev_encoding is not None and prev_action is not None:
        model.adaptation.train()
        tta_optimizer.zero_grad()
        with torch.no_grad():
            out_enc = model(obs_t)
            current_encoding = out_enc["encoding"]
        predicted_encoding = model.adaptation.predict_next_encoding(prev_encoding, prev_action)
        adapt_loss = torch.nn.functional.mse_loss(predicted_encoding, current_encoding.detach())
        adapt_loss.backward()
        tta_optimizer.step()
        model.adaptation.eval()

    use_adapt = tta_optimizer is not None
    with torch.no_grad():
        if hasattr(model, 'adaptation'):
            out = model(obs_t, use_adaptation=use_adapt)
        else:
            out = model(obs_t)

    action     = out["action"].cpu().numpy()[0]
    encoding   = out.get("encoding", None)
    mode_probs = out.get("mode_probs", None)
    if mode_probs is not None:
        mode_probs = mode_probs.cpu().numpy()[0]

    encoding_t = None
    action_t   = None
    if use_adapt and encoding is not None:
        encoding_t = encoding.detach()
        action_t   = out["action"].detach()

    return action, mode_probs, encoding_t, action_t


# ── Per-episode metrics ────────────────────────────────────────────────────────

class EpisodeMetrics:
    """Tracks per-episode recovery metrics after a push is applied."""

    def __init__(self):
        self.push_step    = None    # step when push was applied
        self.peak_tilt    = 0.0    # max tilt angle (radians) after push
        self.tts          = None    # time-to-stabilization (steps) after push
        self.success      = False   # survived episode without falling
        self.made_contact = False   # made useful wall/railing contact
        self.mode_history = []      # per-step mode_probs arrays (post-push only)

    def update(self, step, info, is_post_push, mode_probs=None):
        if is_post_push:
            tilt = info.get("torso_tilt", 0.0)
            self.peak_tilt = max(self.peak_tilt, tilt)
            if self.tts is None and info.get("com_vel", 999) < VELOCITY_STABLE_THRESHOLD:
                self.tts = step - self.push_step
            if mode_probs is not None:
                self.mode_history.append(mode_probs.copy())
        if info.get("useful_contact", False):
            self.made_contact = True


# ── Single-episode rollout ─────────────────────────────────────────────────────

def run_episode(env, model, obs_rms, use_history, history_len, device,
                force_mag=None, push_direction=None, second_push_delay=None,
                use_tta=False):
    """
    Run one episode and return EpisodeMetrics.

    force_mag: if set, overrides the push force magnitude for this episode.
    push_direction: (angle in radians) if set, deterministic push direction.
    second_push_delay: if set, applies a SECOND push this many steps after the first.
    use_tta: if True and model is RecoverFormer, apply test-time adaptation online.
    """
    obs, _ = env.reset()
    history = deque(maxlen=history_len)
    metrics = EpisodeMetrics()
    push_count = 0
    second_push_done = False

    tta_optimizer = None
    prev_encoding = None
    prev_action_t = None
    tta_init_state = None
    if use_tta and use_history:
        tta_init_state = {k: v.clone() for k, v in model.adaptation.state_dict().items()}
        tta_optimizer = torch.optim.Adam(
            model.adaptation.parameters(), lr=1e-3
        )

    for step in range(MAX_EVAL_STEPS):
        action, mode_probs, enc_t, act_t = get_action(
            model, obs_rms, history, obs, use_history, history_len, device,
            tta_optimizer=tta_optimizer,
            prev_encoding=prev_encoding,
            prev_action=prev_action_t,
        )
        prev_encoding = enc_t
        prev_action_t = act_t
        obs, reward, terminated, truncated, info = env.step(action)

        is_pushed = info.get("is_pushed", False)
        if is_pushed:
            push_count += 1
            if push_count == 1 and metrics.push_step is None:
                metrics.push_step = step

        if metrics.push_step is not None:
            metrics.update(step, info, is_post_push=True, mode_probs=mode_probs)

        # Second push injection (sequential push experiment)
        if (second_push_delay is not None
                and metrics.push_step is not None
                and not second_push_done
                and step == metrics.push_step + second_push_delay):
            _inject_push(env, force_mag, push_direction=push_direction,
                         angle_offset=np.pi / 2)  # 90 deg offset from first push
            second_push_done = True

        if terminated or truncated:
            break

    metrics.success = not terminated
    if metrics.tts is None and metrics.success:
        # Survived but never truly stabilized — set TTS to episode length
        metrics.tts = step - (metrics.push_step or 0)

    if tta_init_state is not None:
        model.adaptation.load_state_dict(tta_init_state)
    return metrics


def _inject_push(env, force_mag, push_direction=None, angle_offset=0.0):
    """Directly inject a velocity impulse into the environment."""
    mag = force_mag if force_mag is not None else np.random.uniform(*env.push_force_range)
    angle = (push_direction or np.random.uniform(0, 2 * np.pi)) + angle_offset
    force = np.array([mag * np.cos(angle), mag * np.sin(angle), 0.0])
    impulse_duration = 0.1
    total_mass = max(np.sum(env.model.body_mass), 1.0)
    env.data.qvel[0:3] += force * impulse_duration / total_mass


# ── Experiments ───────────────────────────────────────────────────────────────

def experiment_force_sweep(models_dict, device, n_episodes=100, env_type="open_floor"):
    """
    Table 1: RSR / TTS / PTT at each force magnitude.

    Evaluates each model at 6 force levels: 50, 100, 150, 200, 250, 300N.
    Pushes come from 8 equally-spaced directions, n_episodes total per level.
    """
    force_levels = [50, 100, 150, 200, 250, 300]
    n_directions = 8
    eps_per_dir  = max(1, n_episodes // n_directions)

    print("\n" + "="*80)
    print("EXPERIMENT 1: RSR / TTS / PTT vs Push Force (open floor)")
    print("="*80)
    print(f"{'Force':>6}  {'Model':>14}  {'RSR':>6}  {'TTS(s)':>8}  {'PTT(deg)':>9}")
    print("-"*55)

    results = {}

    for force in force_levels:
        results[force] = {}
        # Reuse one env per force level across all models
        env = G1RecoveryEnv(
            env_type=env_type,
            max_episode_steps=MAX_EVAL_STEPS,
            push_force_range=(force, force),
            push_interval_range=(2.0, 3.0),
        )
        for model_name, (model, obs_rms, use_history, history_len) in models_dict.items():
            rsrs, ttss, ptts, all_modes = [], [], [], []
            for i in range(n_episodes):
                direction = (i % n_directions) * (2 * np.pi / n_directions)
                m = run_episode(env, model, obs_rms, use_history, history_len, device,
                                force_mag=force, push_direction=direction)
                rsrs.append(float(m.success))
                if m.tts is not None:
                    ttss.append(m.tts / 50.0)   # convert steps → seconds
                ptts.append(np.degrees(m.peak_tilt))
                if m.mode_history:
                    all_modes.extend(m.mode_history)

            rsr     = np.mean(rsrs) * 100
            rsr_std = np.std(rsrs) * 100 / np.sqrt(len(rsrs))  # standard error
            tts     = np.mean(ttss) if ttss else float('nan')
            ptt     = np.mean(ptts)

            entry = {"rsr": rsr, "rsr_se": rsr_std, "tts": tts, "ptt": ptt}
            if all_modes:
                avg_mode = np.mean(all_modes, axis=0).tolist()
                entry["avg_mode_probs"] = avg_mode
            results[force][model_name] = entry
            print(f"{force:>5}N  {model_name:>14}  {rsr:>5.1f}%±{rsr_std:>4.1f}  {tts:>7.2f}s  {ptt:>8.1f}°")

        env.close()
        print()

    return results


def experiment_sequential_push(models_dict, device, n_episodes=100,
                                force_mag=150, delay_steps=25):
    """
    Sequential double-push: push at step T, then again at step T+delay.
    delay_steps=25 → 0.5s mid-recovery window.

    This is where history (RecoverFormer) has a structural advantage over MLP.
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: Sequential Double-Push (force={force_mag}N, delay={delay_steps/50:.1f}s)")
    print("="*80)
    print(f"{'Model':>14}  {'Single RSR':>10}  {'Double RSR':>10}  {'Drop':>8}")
    print("-"*50)

    # Pre-generate push angles so all models face the same episodes
    push_angles = [np.random.uniform(0, 2 * np.pi) for _ in range(n_episodes)]
    push_step   = 50   # apply first push at step 50 (1s into episode)

    results = {}
    single_rsrs_all = {m: [] for m in models_dict}
    double_rsrs_all = {m: [] for m in models_dict}

    for mode, rsrs_all in [("single", single_rsrs_all), ("double", double_rsrs_all)]:
        # One env per mode, reused across all models
        env = G1RecoveryEnv(
            env_type="open_floor",
            max_episode_steps=MAX_EVAL_STEPS,
            push_force_range=(force_mag, force_mag),
            push_interval_range=(999.0, 999.0),  # disable auto-push
        )
        for model_name, (model, obs_rms, use_history, history_len) in models_dict.items():
            for ep in range(n_episodes):
                angle = push_angles[ep]
                obs, _ = env.reset()
                history = deque(maxlen=history_len)
                metrics = EpisodeMetrics()
                first_push_done = False
                second_push_done = False

                for step in range(MAX_EVAL_STEPS):
                    action, _, _, _ = get_action(
                        model, obs_rms, history, obs, use_history, history_len, device
                    )
                    obs, reward, terminated, truncated, info = env.step(action)

                    # First push
                    if step == push_step and not first_push_done:
                        _inject_push(env, force_mag, push_direction=angle)
                        metrics.push_step = step
                        first_push_done = True

                    # Second push (90 degrees offset from first)
                    if (mode == "double"
                            and first_push_done
                            and not second_push_done
                            and step == push_step + delay_steps):
                        _inject_push(env, force_mag,
                                     push_direction=angle + np.pi / 2)
                        second_push_done = True

                    if metrics.push_step is not None:
                        metrics.update(step, info, is_post_push=True)

                    if terminated or truncated:
                        break

                metrics.success = not terminated
                rsrs_all[model_name].append(float(metrics.success))

        env.close()

    for model_name in models_dict:
        single_rsrs = single_rsrs_all[model_name]
        double_rsrs = double_rsrs_all[model_name]
        single_rsr = np.mean(single_rsrs) * 100
        double_rsr = np.mean(double_rsrs) * 100
        drop       = single_rsr - double_rsr
        single_se  = np.std(single_rsrs) * 100 / np.sqrt(len(single_rsrs))
        double_se  = np.std(double_rsrs) * 100 / np.sqrt(len(double_rsrs))

        results[model_name] = {
            "single_rsr": single_rsr, "double_rsr": double_rsr, "drop": drop,
            "single_se": single_se, "double_se": double_se,
        }
        print(f"{model_name:>14}  {single_rsr:>5.1f}%±{single_se:>4.1f}  "
              f"{double_rsr:>5.1f}%±{double_se:>4.1f}  {drop:>7.1f}pp")

    return results


def experiment_domain_mismatch(models_dict, device, n_episodes=100, force_mag=150):
    """
    Table 3: RSR under domain mismatch conditions.

    Tests: nominal, low friction, high friction, mass+25%, mass-25%, combined.
    """
    conditions = {
        "Nominal":          dict(env_type="open_floor", friction_range=(0.8, 0.8),  mass_perturbation=0.0,  latency_steps=0),
        "Low friction":     dict(env_type="open_floor", friction_range=(0.3, 0.3),  mass_perturbation=0.0,  latency_steps=0),
        "High friction":    dict(env_type="open_floor", friction_range=(1.5, 1.5),  mass_perturbation=0.0,  latency_steps=0),
        "Mass +25%":        dict(env_type="open_floor", friction_range=(0.8, 0.8),  mass_perturbation=0.25, latency_steps=0),
        "Mass -25%":        dict(env_type="open_floor", friction_range=(0.8, 0.8),  mass_perturbation=-0.2, latency_steps=0),
        "Latency 30ms":     dict(env_type="open_floor", friction_range=(0.8, 0.8),  mass_perturbation=0.0,  latency_steps=2),
        "Combined":         dict(env_type="open_floor", friction_range=(0.3, 0.3),  mass_perturbation=0.2,  latency_steps=2),
        "Walled env":       dict(env_type="walled",     friction_range=(0.8, 0.8),  mass_perturbation=0.0,  latency_steps=0),
    }

    print("\n" + "="*80)
    print(f"EXPERIMENT 3: Domain Mismatch Robustness (force={force_mag}N)")
    print("="*80)

    # Build evaluation entries: standard models + RecoverFormer+TTA variant
    eval_entries = list(models_dict.items())
    rf_entry = next(((n, v) for n, v in models_dict.items() if v[2]), None)
    if rf_entry is not None:
        rf_name, rf_tuple = rf_entry
        eval_entries.append((rf_name + "+TTA", rf_tuple))

    header = f"{'Condition':<16}" + "".join(f"  {n:>14}" for n, _ in eval_entries)
    print(header)
    print("-" * len(header))

    results = {}
    for cond_name, env_kwargs in conditions.items():
        results[cond_name] = {}
        row = f"{cond_name:<16}"
        # Reuse one env per condition across all models
        kw = dict(env_kwargs)
        env_type = kw.pop("env_type", "open_floor")
        env = G1RecoveryEnv(
            env_type=env_type,
            max_episode_steps=MAX_EVAL_STEPS,
            push_force_range=(force_mag, force_mag),
            push_interval_range=(2.0, 3.0),
            **kw,
        )
        for model_name, (model, obs_rms, use_history, history_len) in eval_entries:
            use_tta = model_name.endswith("+TTA")
            rsrs = []
            for _ in range(n_episodes):
                m = run_episode(env, model, obs_rms, use_history, history_len, device,
                                force_mag=force_mag, use_tta=use_tta)
                rsrs.append(float(m.success))

            rsr    = np.mean(rsrs) * 100
            rsr_se = np.std(rsrs) * 100 / np.sqrt(len(rsrs))
            results[cond_name][model_name] = {"rsr": rsr, "rsr_se": rsr_se}
            row += f"  {rsr:>11.1f}%±{rsr_se:.1f}"
        env.close()
        print(row)

    return results


def experiment_contact_aware(models_dict, device, n_episodes=100, env_type="walled"):
    """
    Table 2: Contact-aware recovery in walled/cluttered environments.

    Metrics: RSR + Contact Efficiency Rate (CER).
    """
    force_levels = [100, 200, 300]

    print("\n" + "="*80)
    print(f"EXPERIMENT 2: Contact-Aware Recovery ({env_type})")
    print("="*80)
    print(f"{'Force':>6}  {'Model':>14}  {'RSR':>12}  {'CER':>6}")
    print("-"*48)

    results = {}

    for force in force_levels:
        results[force] = {}
        # Reuse one env per force level (env.reset() reinitializes push timer)
        env = G1RecoveryEnv(
            env_type=env_type,
            max_episode_steps=MAX_EVAL_STEPS,
            push_force_range=(force, force),
            push_interval_range=(2.0, 3.0),
        )
        for model_name, (model, obs_rms, use_history, history_len) in models_dict.items():
            rsrs, contact_and_success = [], []
            for _ in range(n_episodes):
                m = run_episode(env, model, obs_rms, use_history, history_len, device,
                                force_mag=force)
                rsrs.append(float(m.success))
                contact_and_success.append(float(m.made_contact and m.success))

            rsr     = np.mean(rsrs) * 100
            rsr_se  = np.std(rsrs) * 100 / np.sqrt(len(rsrs))
            n_succ  = sum(rsrs)
            # CER: among successful recoveries, fraction that used contact
            cer = (sum(contact_and_success) / n_succ * 100) if n_succ > 0 else 0.0
            results[force][model_name] = {"rsr": rsr, "rsr_se": rsr_se, "cer": cer}
            print(f"{force:>5}N  {model_name:>14}  {rsr:>5.1f}%±{rsr_se:>4.1f}  {cer:>5.1f}%")
        env.close()
        print()

    return results


def experiment_walled_force_sweep(models_dict, device, n_episodes=100):
    """
    Zero-shot walled env force sweep: open-floor-trained models in walled env.

    Key comparison: RF (zero-shot) vs Baseline (zero-shot) in walled env at
    multiple force levels. Expected: RF ≈ 100% all forces, Baseline ≈ 0%.
    """
    force_levels = [100, 150, 200, 250, 300]

    print("\n" + "="*80)
    print("EXPERIMENT: Zero-Shot Walled Env Force Sweep")
    print("(open-floor-trained models evaluated in walled environment)")
    print("="*80)
    print(f"{'Force':>6}  {'Model':>14}  {'RSR':>12}  {'CER':>6}")
    print("-"*48)

    results = {}
    for force in force_levels:
        results[force] = {}
        env = G1RecoveryEnv(
            env_type="walled",
            max_episode_steps=MAX_EVAL_STEPS,
            push_force_range=(force, force),
            push_interval_range=(2.0, 3.0),
        )
        for model_name, (model, obs_rms, use_history, history_len) in models_dict.items():
            rsrs, contact_and_success = [], []
            for _ in range(n_episodes):
                m = run_episode(env, model, obs_rms, use_history, history_len, device,
                                force_mag=force)
                rsrs.append(float(m.success))
                contact_and_success.append(float(m.made_contact and m.success))
            rsr    = np.mean(rsrs) * 100
            rsr_se = np.std(rsrs) * 100 / np.sqrt(len(rsrs))
            n_succ = sum(rsrs)
            cer    = (sum(contact_and_success) / n_succ * 100) if n_succ > 0 else 0.0
            results[force][model_name] = {"rsr": rsr, "rsr_se": rsr_se, "cer": cer}
            print(f"{force:>5}N  {model_name:>14}  {rsr:>5.1f}%±{rsr_se:>4.1f}  {cer:>5.1f}%")
        env.close()
        print()
    return results


def experiment_wall_distance_sweep(models_dict, device, n_episodes=100, force_mag=150):
    """
    Wall distance sweep: vary wall distance from 0.25m to 1.4m.

    Shows RF gracefully degrades as wall becomes unreachable (affordance head
    guides contact when possible, falls back to open-floor recovery when not).
    Baseline fails at close distances (crashes into wall) and also fails at
    medium distances (can't avoid or exploit).
    """
    distances = [0.25, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20, 1.40]

    print("\n" + "="*80)
    print(f"EXPERIMENT: Wall Distance Sweep (force={force_mag}N)")
    print("(open-floor-trained models; wall_distance varied per condition)")
    print("="*80)
    print(f"{'Dist(m)':>8}  {'Model':>14}  {'RSR':>12}  {'CER':>6}")
    print("-"*50)

    results = {}
    for dist in distances:
        results[dist] = {}
        env = G1RecoveryEnv(
            env_type="walled",
            max_episode_steps=MAX_EVAL_STEPS,
            push_force_range=(force_mag, force_mag),
            push_interval_range=(2.0, 3.0),
            wall_distance=dist,
        )
        for model_name, (model, obs_rms, use_history, history_len) in models_dict.items():
            rsrs, contact_and_success = [], []
            for _ in range(n_episodes):
                m = run_episode(env, model, obs_rms, use_history, history_len, device,
                                force_mag=force_mag)
                rsrs.append(float(m.success))
                contact_and_success.append(float(m.made_contact and m.success))
            rsr    = np.mean(rsrs) * 100
            rsr_se = np.std(rsrs) * 100 / np.sqrt(len(rsrs))
            n_succ = sum(rsrs)
            cer    = (sum(contact_and_success) / n_succ * 100) if n_succ > 0 else 0.0
            results[dist][model_name] = {"rsr": rsr, "rsr_se": rsr_se, "cer": cer}
            print(f"{dist:>7.2f}m  {model_name:>14}  {rsr:>5.1f}%±{rsr_se:>4.1f}  {cer:>5.1f}%")
        env.close()
        print()
    return results


def experiment_push_direction(models_dict, device, n_episodes=100, force_mag=150):
    """
    Push direction analysis in walled env: toward wall vs. away from wall.

    Wall is at +x (angle=0). Toward wall = push in +x direction.
    Away from wall = push in -x direction. Lateral = ±y.

    RF should exploit wall when pushed toward it (CER high), recover normally
    when pushed away. Baseline fails when pushed toward wall, succeeds otherwise.
    """
    # Wall is at +x. Push direction = direction robot is pushed.
    # Push toward +x wall means push force in +x direction → robot moves toward wall.
    direction_configs = {
        "Toward wall (+x)":  0.0,               # push toward front wall
        "Lateral (+y)":      np.pi / 2,          # push toward left wall
        "Away from wall":    np.pi,              # push away from front wall
        "Lateral (-y)":      3 * np.pi / 2,     # push away from left wall
    }

    print("\n" + "="*80)
    print(f"EXPERIMENT: Push Direction Analysis in Walled Env (force={force_mag}N)")
    print("(open-floor-trained models; wall at +x and +y, distance=0.5m)")
    print("="*80)
    print(f"{'Direction':>22}  {'Model':>14}  {'RSR':>12}  {'CER':>6}")
    print("-"*60)

    results = {}
    env = G1RecoveryEnv(
        env_type="walled",
        max_episode_steps=MAX_EVAL_STEPS,
        push_force_range=(force_mag, force_mag),
        push_interval_range=(999.0, 999.0),  # disable auto-push; we control direction
        wall_distance=0.5,
    )

    for dir_name, angle in direction_configs.items():
        results[dir_name] = {}
        for model_name, (model, obs_rms, use_history, history_len) in models_dict.items():
            rsrs, contact_and_success = [], []
            for _ in range(n_episodes):
                m = run_episode(env, model, obs_rms, use_history, history_len, device,
                                force_mag=force_mag, push_direction=angle)
                rsrs.append(float(m.success))
                contact_and_success.append(float(m.made_contact and m.success))
            rsr    = np.mean(rsrs) * 100
            rsr_se = np.std(rsrs) * 100 / np.sqrt(len(rsrs))
            n_succ = sum(rsrs)
            cer    = (sum(contact_and_success) / n_succ * 100) if n_succ > 0 else 0.0
            results[dir_name][model_name] = {"rsr": rsr, "rsr_se": rsr_se, "cer": cer}
            print(f"{dir_name:>22}  {model_name:>14}  {rsr:>5.1f}%±{rsr_se:>4.1f}  {cer:>5.1f}%")
        print()
    env.close()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # flush on every newline when piped

    parser = argparse.ArgumentParser(description="RecoverFormer Evaluation")
    parser.add_argument("--recoverformer", type=str, required=True,
                        help="Path to RecoverFormer checkpoint (supports glob)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline MLP checkpoint (supports glob)")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["force_sweep", "sequential_push",
                                 "domain_mismatch", "contact_aware",
                                 "walled_force_sweep", "wall_distance_sweep",
                                 "push_direction"])
    parser.add_argument("--env_type", type=str, default="open_floor",
                        choices=["open_floor", "walled", "cluttered"])
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--force_mag", type=float, default=150,
                        help="Fixed force for sequential/mismatch experiments")
    parser.add_argument("--double_push_delay", type=float, default=0.5,
                        help="Seconds between first and second push (sequential experiment)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override results output directory (default: ../results)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve glob paths
    def resolve_ckpt(path_str):
        p = Path(path_str)
        if "*" in str(p):
            matches = sorted(p.parent.glob(p.name))
            if not matches:
                raise FileNotFoundError(f"No checkpoint found: {path_str}")
            return matches[-1]
        return p

    rf_ckpt = resolve_ckpt(args.recoverformer)
    print(f"RecoverFormer: {rf_ckpt}")

    models_dict = {}
    m, name, rms, uh, hl = load_model(str(rf_ckpt), device)
    models_dict["RecoverFormer"] = (m, rms, uh, hl)
    # Ablation: same RF weights but history_len=1 — isolates benefit of temporal context
    models_dict["RF-no-hist"] = (m, rms, True, 1)

    if args.baseline:
        bl_ckpt = resolve_ckpt(args.baseline)
        print(f"Baseline MLP:  {bl_ckpt}")
        m, name, rms, uh, hl = load_model(str(bl_ckpt), device)
        models_dict["Baseline MLP"] = (m, rms, uh, hl)

    delay_steps = int(args.double_push_delay * 50)  # convert seconds to steps

    results_dir = Path(args.results_dir) if args.results_dir else Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "force_sweep":
        results = experiment_force_sweep(models_dict, device, args.n_episodes, args.env_type)
        # Convert int keys to str for JSON
        out = {str(k): v for k, v in results.items()}
        with open(results_dir / "force_sweep.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {results_dir / 'force_sweep.json'}")

    elif args.experiment == "sequential_push":
        results = experiment_sequential_push(models_dict, device, args.n_episodes,
                                   force_mag=args.force_mag,
                                   delay_steps=delay_steps)
        fname = f"sequential_push_{int(args.force_mag)}N_{int(delay_steps)}steps.json"
        with open(results_dir / fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_dir / fname}")

    elif args.experiment == "domain_mismatch":
        results = experiment_domain_mismatch(models_dict, device, args.n_episodes,
                                   force_mag=args.force_mag)
        with open(results_dir / "domain_mismatch.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_dir / 'domain_mismatch.json'}")

    elif args.experiment == "contact_aware":
        results = experiment_contact_aware(models_dict, device, args.n_episodes,
                                 env_type=args.env_type)
        out = {str(k): v for k, v in results.items()}
        with open(results_dir / "contact_aware.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {results_dir / 'contact_aware.json'}")

    elif args.experiment == "walled_force_sweep":
        results = experiment_walled_force_sweep(models_dict, device, args.n_episodes)
        out = {str(k): v for k, v in results.items()}
        with open(results_dir / "walled_force_sweep.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {results_dir / 'walled_force_sweep.json'}")

    elif args.experiment == "wall_distance_sweep":
        results = experiment_wall_distance_sweep(models_dict, device, args.n_episodes,
                                                 force_mag=args.force_mag)
        out = {str(k): v for k, v in results.items()}
        with open(results_dir / "wall_distance_sweep.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {results_dir / 'wall_distance_sweep.json'}")

    elif args.experiment == "push_direction":
        results = experiment_push_direction(models_dict, device, args.n_episodes,
                                            force_mag=args.force_mag)
        with open(results_dir / "push_direction.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_dir / 'push_direction.json'}")


if __name__ == "__main__":
    main()
