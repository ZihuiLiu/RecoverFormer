"""
Baseline MLP policy for comparison (PPO-Flat / PPO-DR baselines).
Standard feedforward actor-critic with no mode head, affordance, or adaptation.
"""

import torch
import torch.nn as nn
import numpy as np


class BaselineMLP(nn.Module):
    """Simple MLP actor-critic for baseline comparisons."""

    def __init__(self, obs_dim: int = 106, action_dim: int = 29, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Action log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> dict:
        """
        Args:
            obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
        Returns:
            dict with action, value
        """
        # If given history, just use the last observation
        if obs.dim() == 3:
            obs = obs[:, -1, :]

        action = self.actor(obs)
        value = self.critic(obs).squeeze(-1)

        return {
            "action": action,
            "value": value,
            "mode_probs": None,
            "affordances": None,
            "encoding": None,
        }

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            out = self.forward(obs)
        return out["action"].cpu().numpy()
