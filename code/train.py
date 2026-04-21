"""
PPO Training script for RecoverFormer.

Usage:
    python train.py --model recoverformer --env_type open_floor --num_envs 16
    python train.py --model baseline --env_type open_floor --num_envs 16
    python train.py --model recoverformer --env_type walled --num_envs 16
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from envs.g1_recovery_env import G1RecoveryEnv, OBS_DIM_BASE, N_ACTUATORS
from models.recoverformer import RecoverFormer
from models.baseline_mlp import BaselineMLP


class RunningMeanStd:
    """Tracks running mean and variance for observation normalization."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean.astype(np.float32)) / (np.sqrt(self.var).astype(np.float32) + 1e-8)


class RolloutBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self, n_envs: int, n_steps: int, obs_dim: int, action_dim: int,
                 history_len: int = 50, device: str = "cpu", use_history: bool = False):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.device = device
        self.use_history = use_history

        # Buffers
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Observation histories for transformer-based policies
        if use_history:
            self.obs_histories = np.zeros(
                (n_steps, n_envs, history_len, obs_dim), dtype=np.float32
            )

        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob, obs_history=None):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        if self.use_history and obs_history is not None:
            self.obs_histories[idx] = obs_history
        self.ptr += 1

    def compute_returns(self, last_values: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def reset(self):
        self.ptr = 0


class PPOTrainer:
    """PPO trainer for G1 recovery policies."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # GPU power management: reduce peak load to prevent system crashes
        if self.device.type == "cuda":
            # Limit GPU memory to avoid power spikes from large allocations
            if args.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
                print(f"GPU memory limited to {args.gpu_memory_fraction*100:.0f}%")
            # Enable TF32 for lower power draw on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled (lower power, minimal accuracy impact)")

        # Create environments
        self.envs = []
        for i in range(args.num_envs):
            _lat_range = (0, args.latency_max_steps) if args.latency_max_steps > 0 else None
            env = G1RecoveryEnv(
                env_type=args.env_type,
                max_episode_steps=args.max_episode_steps,
                push_force_range=(args.push_min, args.push_max),
                push_interval_range=(args.push_interval_min, args.push_interval_max),
                friction_range=(args.friction_min, args.friction_max),
                mass_perturbation=args.mass_perturbation,
                latency_steps_range=_lat_range,
            )
            self.envs.append(env)

        # Live viewer for env 0 (optional)
        self.viewer = None
        if args.render:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(
                self.envs[0].model, self.envs[0].data
            )
            print("Live viewer opened for environment 0")

        self.n_envs = args.num_envs
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.action_dim = N_ACTUATORS
        print(f"obs_dim={self.obs_dim}, action_dim={self.action_dim}")

        # Create model
        if args.model == "recoverformer":
            self.policy = RecoverFormer(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                embed_dim=args.embed_dim,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                n_modes=args.n_modes,
                history_len=args.history_len,
            ).to(self.device)
            self.use_history = True
        else:
            self.policy = BaselineMLP(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
            ).to(self.device)
            self.use_history = False

        # Action distribution (Gaussian)
        # init_noise_std=0.3 — with action_scale=0.25, this gives 0.075 rad noise
        # (unitree_rl_gym uses 0.8 but with 4096 envs; we have 64, so start lower)
        self.action_log_std = nn.Parameter(
            torch.full((self.action_dim,), np.log(args.init_std), device=self.device)
        )

        # Optimizer
        params = list(self.policy.parameters()) + [self.action_log_std]
        self.optimizer = optim.Adam(params, lr=args.lr, eps=1e-5)

        # Finetune: load only model weights (keep fresh optimizer + reset step count).
        # Supports obs-dim extension: if the checkpoint's input-projection accepts
        # fewer obs dims than current model, zero-pad the extra channels so the
        # loaded policy behaves identically on the original obs and has zero
        # initial response to the new obs channels (e.g., velocity command).
        if getattr(args, "finetune", None):
            ckpt = torch.load(args.finetune, map_location=self.device, weights_only=False)
            ckpt_state = ckpt["model_state_dict"]
            cur_state = self.policy.state_dict()
            remapped = {}
            for key, ckpt_w in ckpt_state.items():
                if key in cur_state and cur_state[key].shape != ckpt_w.shape:
                    cur_w = cur_state[key]
                    if cur_w.dim() == 2 and cur_w.shape[0] == ckpt_w.shape[0] \
                       and cur_w.shape[1] > ckpt_w.shape[1]:
                        # Input-dim was extended: zero-pad new input columns.
                        new_w = cur_w.clone()
                        new_w[:, :ckpt_w.shape[1]] = ckpt_w
                        new_w[:, ckpt_w.shape[1]:] = 0.0
                        remapped[key] = new_w
                        print(f"[Finetune] Zero-padded '{key}': "
                              f"{tuple(ckpt_w.shape)} -> {tuple(cur_w.shape)}")
                    else:
                        print(f"[Finetune] Shape mismatch on '{key}' "
                              f"({tuple(ckpt_w.shape)} vs {tuple(cur_w.shape)}); skipping")
                else:
                    remapped[key] = ckpt_w
            missing, unexpected = self.policy.load_state_dict(remapped, strict=False)
            if missing:
                print(f"[Finetune] Missing keys: {missing}")
            if unexpected:
                print(f"[Finetune] Unexpected keys: {unexpected}")
            if "action_log_std" in ckpt:
                self.action_log_std.data = ckpt["action_log_std"]
            print(f"[Finetune] Loaded weights from {args.finetune}")

        # Rollout buffer
        self.buffer = RolloutBuffer(
            self.n_envs, args.n_steps, self.obs_dim, self.action_dim,
            history_len=args.history_len, device=str(self.device),
            use_history=self.use_history,
        )

        # History buffers per environment
        if self.use_history:
            self.history_buffers = [
                deque(maxlen=args.history_len) for _ in range(self.n_envs)
            ]

        # Logging
        self.log_dir = Path(args.log_dir) / f"{args.model}_{args.env_type}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

        # Save config
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        # Observation normalization
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))

        # Stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_steps = 0

        # Per-env episode accumulators (persistent across rollouts)
        self._ep_rewards = [0.0] * self.n_envs
        self._ep_lengths = [0] * self.n_envs

        print(f"Model: {args.model} ({sum(p.numel() for p in self.policy.parameters()):,} parameters)")
        print(f"Environments: {self.n_envs} x {args.env_type}")
        print(f"Log dir: {self.log_dir}")

    def _normalize_obs(self, obs_array):
        """Normalize observations using running statistics."""
        return self.obs_rms.normalize(obs_array)

    def _get_obs_tensor(self, obs_list):
        """Convert list of observations to tensor, with history if needed."""
        if self.use_history:
            # Build history tensors for each env
            histories = []
            for i, obs in enumerate(obs_list):
                norm_obs = self._normalize_obs(obs)
                self.history_buffers[i].append(norm_obs)
                # Pad if history too short
                hist = list(self.history_buffers[i])
                while len(hist) < self.args.history_len:
                    hist.insert(0, hist[0])
                histories.append(np.stack(hist))
            return torch.tensor(np.stack(histories), dtype=torch.float32, device=self.device)
        else:
            norm_obs = self._normalize_obs(np.stack(obs_list))
            return torch.tensor(norm_obs, dtype=torch.float32, device=self.device)

    def _sample_action(self, obs_tensor):
        """Sample action from policy with Gaussian noise."""
        with torch.no_grad():
            out = self.policy(obs_tensor)
            mean = out["action"]
            std = self.action_log_std.exp().clamp(min=0.02, max=0.5)

            # Sample from Gaussian, clamp, THEN compute log_prob on clamped action
            # (must match update phase which computes log_prob on stored clamped actions)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.cpu().numpy(),
            out["value"].cpu().numpy(),
            log_prob.cpu().numpy(),
            out.get("mode_probs"),
        )

    def collect_rollout(self, obs_list=None):
        """Collect n_steps of experience from all environments."""
        self.buffer.reset()

        if obs_list is None:
            obs_list = [None] * self.n_envs

        # Only reset envs that don't have a current observation (first call or after init)
        for i, env in enumerate(self.envs):
            if obs_list[i] is None:
                obs, _ = env.reset()
                obs_list[i] = obs
                self._ep_rewards[i] = 0.0
                self._ep_lengths[i] = 0
                if self.use_history:
                    self.history_buffers[i].clear()

        for step in range(self.args.n_steps):
            # Update observation normalization stats with raw observations
            self.obs_rms.update(np.stack(obs_list))
            obs_tensor = self._get_obs_tensor(obs_list)
            actions, values, log_probs, mode_probs = self._sample_action(obs_tensor)

            # Step all environments
            next_obs_list = [None] * self.n_envs
            rewards = np.zeros(self.n_envs, dtype=np.float32)
            dones = np.zeros(self.n_envs, dtype=np.float32)

            for i, env in enumerate(self.envs):
                next_obs, reward, terminated, truncated, info = env.step(actions[i])
                rewards[i] = reward * self.args.reward_scale
                done = terminated or truncated
                dones[i] = float(done)

                self._ep_rewards[i] += reward
                self._ep_lengths[i] += 1

                if done:
                    self.episode_rewards.append(self._ep_rewards[i])
                    self.episode_lengths.append(self._ep_lengths[i])
                    self._ep_rewards[i] = 0.0
                    self._ep_lengths[i] = 0
                    next_obs, _ = env.reset()
                    if self.use_history:
                        self.history_buffers[i].clear()

                next_obs_list[i] = next_obs

            # Sync live viewer (env 0)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

            # Store obs history for transformer (already normalized)
            obs_history_np = obs_tensor.cpu().numpy() if self.use_history else None

            self.buffer.add(
                np.stack(obs_list), actions, rewards, dones, values, log_probs,
                obs_history=obs_history_np,
            )
            obs_list = next_obs_list
            self.total_steps += self.n_envs

        # Compute last values for GAE
        with torch.no_grad():
            last_obs_tensor = self._get_obs_tensor(obs_list)
            last_out = self.policy(last_obs_tensor)
            last_values = last_out["value"].cpu().numpy()

        self.buffer.compute_returns(last_values, gamma=self.args.gamma, gae_lambda=self.args.gae_lambda)

        return obs_list  # return for continuity

    def update(self):
        """PPO update step."""
        # Flatten buffer
        B = self.args.n_steps * self.n_envs

        # Use observation histories for transformer-based policies
        if self.use_history:
            # Already normalized during rollout collection
            obs_flat = self.buffer.obs_histories.reshape(
                B, self.args.history_len, self.obs_dim
            )
        else:
            obs_flat = self._normalize_obs(self.buffer.obs.reshape(B, self.obs_dim))

        actions_flat = self.buffer.actions.reshape(B, self.action_dim)
        old_log_probs_flat = self.buffer.log_probs.reshape(B)
        advantages_flat = self.buffer.advantages.reshape(B)
        returns_flat = self.buffer.returns.reshape(B)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Convert to tensors
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions_flat, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs_flat, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages_flat, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns_flat, dtype=torch.float32, device=self.device)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_mode_loss = 0
        total_approx_kl = 0
        n_updates = 0
        kl_exceeded = False

        for epoch in range(self.args.n_epochs):
            if kl_exceeded:
                break

            # Mini-batch indices
            indices = np.random.permutation(B)
            for start in range(0, B, self.args.batch_size):
                end = min(start + self.args.batch_size, B)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                # Forward pass
                out = self.policy(mb_obs)
                mean = out["action"]
                std = self.action_log_std.exp().clamp(min=0.02, max=0.5)

                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # KL early stopping: abort if policy changed too much
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                if self.args.target_kl is not None and abs(approx_kl) > self.args.target_kl:
                    kl_exceeded = True
                    break

                # PPO clipped objective
                log_ratio = new_log_probs - mb_old_log_probs
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                ratio = log_ratio.exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = out["value"]
                value_loss = F.mse_loss(value_pred, mb_returns)

                # Mode specialization + dead-mode penalty (RecoverFormer only)
                mode_loss = torch.tensor(0.0, device=self.device)
                if out["mode_probs"] is not None:
                    mode_probs = out["mode_probs"]
                    # Step-level: low entropy → decisive per-step mode
                    step_entropy_loss = self.policy.compute_mode_loss(mode_probs)
                    # Batch-level: penalize modes that are completely unused
                    # (one-sided — only penalizes underuse, not overuse, so it
                    #  doesn't conflict with step-level decisiveness)
                    avg_usage = mode_probs.mean(dim=0)  # (n_modes,)
                    min_usage = 1.0 / mode_probs.shape[-1] * 0.4  # target: each mode ≥40% of 1/K
                    dead_penalty = torch.relu(min_usage - avg_usage).sum()
                    mode_loss = step_entropy_loss + dead_penalty

                # Total loss
                loss = (
                    policy_loss
                    + self.args.vf_coef * value_loss
                    - self.args.ent_coef * entropy
                    + self.args.mode_coef * mode_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                all_params = list(self.policy.parameters()) + [self.action_log_std]
                nn.utils.clip_grad_norm_(all_params, self.args.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
                total_mode_loss += mode_loss.item()
                total_approx_kl += abs(approx_kl)
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy_loss / n_updates,
            "mode_loss": total_mode_loss / n_updates,
            "approx_kl": total_approx_kl / n_updates,
        }

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.total_timesteps:,} steps...")
        start_time = time.time()

        obs_list = None
        iteration = 0

        while self.total_steps < self.args.total_timesteps:
            iteration += 1

            # Anneal learning rate linearly
            progress = self.total_steps / self.args.total_timesteps
            lr_now = self.args.lr * (1.0 - progress)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_now

            # Anneal temperature for RecoverFormer
            if hasattr(self.policy, 'anneal_temperature'):
                self.policy.anneal_temperature(progress)

            # Collect rollout (carry obs_list across rollouts for episode continuity)
            obs_list = self.collect_rollout(obs_list)

            # PPO update
            losses = self.update()

            # Logging
            elapsed = time.time() - start_time
            fps = self.total_steps / elapsed if elapsed > 0 else 0

            if iteration % self.args.log_interval == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                current_std = self.action_log_std.exp().clamp(min=0.02, max=0.5).mean().item()

                print(f"Iter {iteration:5d} | Steps {self.total_steps:10,d} | "
                      f"FPS {fps:6.0f} | "
                      f"Reward {mean_reward:8.2f} | "
                      f"EpLen {mean_length:6.1f} | "
                      f"Std {current_std:.4f} | "
                      f"KL {losses['approx_kl']:.4f} | "
                      f"PL {losses['policy_loss']:.4f} | "
                      f"VL {losses['value_loss']:.4f} | "
                      f"Ent {losses['entropy']:.4f}")

                self.writer.add_scalar("reward/mean", mean_reward, self.total_steps)
                self.writer.add_scalar("reward/ep_length", mean_length, self.total_steps)
                self.writer.add_scalar("loss/policy", losses["policy_loss"], self.total_steps)
                self.writer.add_scalar("loss/value", losses["value_loss"], self.total_steps)
                self.writer.add_scalar("loss/entropy", losses["entropy"], self.total_steps)
                self.writer.add_scalar("loss/mode", losses["mode_loss"], self.total_steps)
                self.writer.add_scalar("loss/approx_kl", losses["approx_kl"], self.total_steps)
                self.writer.add_scalar("param/action_std", current_std, self.total_steps)
                self.writer.add_scalar("perf/fps", fps, self.total_steps)

                if hasattr(self.policy, 'temperature'):
                    self.writer.add_scalar("param/temperature",
                                          self.policy.temperature.item(), self.total_steps)

            # Save checkpoint
            if iteration % self.args.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{iteration}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        self.writer.close()
        if self.viewer is not None:
            self.viewer.close()
        print(f"\nTraining complete! Total time: {time.time() - start_time:.1f}s")
        print(f"Model saved to: {self.log_dir}")

    def save_checkpoint(self, filename: str):
        path = self.log_dir / filename
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "action_log_std": self.action_log_std.data,
            "total_steps": self.total_steps,
            "args": vars(self.args),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.action_log_std.data = ckpt["action_log_std"]
        self.total_steps = ckpt["total_steps"]


def parse_args():
    parser = argparse.ArgumentParser(description="RecoverFormer PPO Training")

    # Model
    parser.add_argument("--model", type=str, default="recoverformer",
                        choices=["recoverformer", "baseline"])
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_modes", type=int, default=4)
    parser.add_argument("--history_len", type=int, default=50)

    # Environment
    parser.add_argument("--env_type", type=str, default="open_floor",
                        choices=["open_floor", "walled", "cluttered"])
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--push_min", type=float, default=50.0)
    parser.add_argument("--push_max", type=float, default=200.0)
    parser.add_argument("--push_interval_min", type=float, default=1.0)
    parser.add_argument("--push_interval_max", type=float, default=3.0)
    parser.add_argument("--friction_min", type=float, default=0.5)
    parser.add_argument("--friction_max", type=float, default=1.2)
    parser.add_argument("--mass_perturbation", type=float, default=0.0)
    parser.add_argument("--latency_max_steps", type=int, default=0,
                        help="Max steps of action latency to randomize per episode (0 disables)")
    parser.add_argument("--init_std", type=float, default=0.3,
                        help="Initial action Gaussian std")

    # PPO
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--n_steps", type=int, default=48)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--mode_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_kl", type=float, default=None,
                        help="KL early stopping threshold (None to disable)")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)

    # GPU power management
    parser.add_argument("--gpu_memory_fraction", type=float, default=1.0,
                        help="Limit GPU memory usage (0.0-1.0). Use 0.7 to reduce power spikes.")
    parser.add_argument("--gpu_power_save", action="store_true",
                        help="Enable power-saving mode: smaller batch, fewer envs, memory cap.")
    parser.add_argument("--render", action="store_true",
                        help="Open live MuJoCo viewer to watch env 0 during training.")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to checkpoint for weights-only loading (resets optimizer and step count).")

    args = parser.parse_args()

    # Reduce batch size for transformer to fit in GPU memory
    if args.model == "recoverformer" and args.batch_size > 128:
        args.batch_size = 128
        print(f"[RecoverFormer] batch_size reduced to {args.batch_size} (transformer + history)")

    # Apply power-save preset: reduces GPU load to prevent system crashes
    if args.gpu_power_save:
        args.num_envs = min(args.num_envs, 64)
        args.batch_size = min(args.batch_size, 256)
        args.gpu_memory_fraction = min(args.gpu_memory_fraction, 0.7)
        print(f"[Power-save] num_envs={args.num_envs}, batch_size={args.batch_size}, GPU memory=70%")

    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = PPOTrainer(args)
    trainer.train()
