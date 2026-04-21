"""
RecoverFormer: Transformer-based humanoid recovery policy.

Architecture:
  1. Observation Encoder: Causal transformer over observation history
  2. Latent Recovery Mode Head: Gumbel-Softmax discrete mode prediction
  3. Contact Affordance Head: Sigmoid prediction of contact region values
  4. Action Decoder: MLP conditioned on encoding, mode, and affordances
  5. Test-Time Adaptation Module: Small residual MLP updated online
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class CausalTransformerEncoder(nn.Module):
    """Causal transformer encoder for observation history."""

    def __init__(self, obs_dim: int, embed_dim: int = 256, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        # Project observation to embedding
        self.obs_proj = nn.Linear(obs_dim, embed_dim)

        # Learnable positional encoding
        self.pos_embed = nn.Embedding(128, embed_dim)  # up to 128 timesteps

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_history: (batch, seq_len, obs_dim) observation history
        Returns:
            encoding: (batch, embed_dim) encoding of current timestep
        """
        B, T, _ = obs_history.shape

        # Project and add positional encoding
        x = self.obs_proj(obs_history)  # (B, T, embed_dim)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Transformer forward
        x = self.transformer(x, mask=causal_mask)
        x = self.layer_norm(x)

        # Return encoding of the last (current) timestep
        return x[:, -1, :]  # (B, embed_dim)


class LatentRecoveryModeHead(nn.Module):
    """Predicts a discrete latent recovery mode using Gumbel-Softmax."""

    def __init__(self, embed_dim: int = 256, n_modes: int = 4, mode_embed_dim: int = 32):
        super().__init__()
        self.n_modes = n_modes
        self.mode_embed_dim = mode_embed_dim

        self.logit_proj = nn.Linear(embed_dim, n_modes)
        self.mode_embedding = nn.Embedding(n_modes, mode_embed_dim)

    def forward(self, encoding: torch.Tensor, temperature: float = 1.0,
                hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoding: (batch, embed_dim)
            temperature: Gumbel-Softmax temperature (scales softmax sharpness)
            hard: if True, use hard argmax at inference
        Returns:
            mode_embed: (batch, mode_embed_dim)
            mode_probs: (batch, n_modes) soft mode probabilities
        """
        logits = self.logit_proj(encoding)  # (B, n_modes)
        # Temperature-scaled softmax (deterministic — Gumbel noise breaks PPO ratios)
        mode_probs = F.softmax(logits / max(temperature, 0.01), dim=-1)

        if not self.training:
            # Hard argmax at inference
            idx = torch.argmax(logits, dim=-1)
            mode_onehot = F.one_hot(idx, self.n_modes).float()
            mode_embed = mode_onehot @ self.mode_embedding.weight
        else:
            # Soft weighted sum of mode embeddings (differentiable, deterministic)
            mode_embed = mode_probs @ self.mode_embedding.weight  # (B, mode_embed_dim)

        return mode_embed, mode_probs


class ContactAffordanceHead(nn.Module):
    """Predicts stabilization value of K_c candidate contact regions."""

    def __init__(self, embed_dim: int = 256, n_regions: int = 8):
        super().__init__()
        self.n_regions = n_regions
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_regions),
            nn.Sigmoid(),
        )

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoding: (batch, embed_dim)
        Returns:
            affordances: (batch, n_regions) values in [0, 1]
        """
        return self.net(encoding)


class ActionDecoder(nn.Module):
    """MLP action decoder conditioned on encoding, mode, and affordances."""

    def __init__(self, embed_dim: int = 256, mode_embed_dim: int = 32,
                 n_regions: int = 8, action_dim: int = 29):
        super().__init__()
        input_dim = embed_dim + mode_embed_dim + n_regions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, encoding: torch.Tensor, mode_embed: torch.Tensor,
                affordances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoding: (batch, embed_dim)
            mode_embed: (batch, mode_embed_dim)
            affordances: (batch, n_regions)
        Returns:
            action: (batch, action_dim) in [-1, 1]
        """
        x = torch.cat([encoding, mode_embed, affordances], dim=-1)
        return self.net(x)


class AdaptationModule(nn.Module):
    """Lightweight test-time adaptation module.

    Predicts residual action corrections. At test time, the dynamics predictor
    is used to compute an adaptation loss, and only this module's parameters
    are updated online.
    """

    def __init__(self, embed_dim: int = 256, action_dim: int = 29, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim

        # Residual action predictor
        self.residual_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Forward dynamics predictor (for self-supervised adaptation loss)
        self.dynamics_net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Scale the residual to be small initially
        with torch.no_grad():
            self.residual_net[-2].weight.mul_(0.01)
            self.residual_net[-2].bias.zero_()

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """Predict residual action correction."""
        return self.residual_net(encoding) * 0.1  # scale down residual

    def predict_next_encoding(self, encoding: torch.Tensor,
                               action: torch.Tensor) -> torch.Tensor:
        """Predict next-step encoding for adaptation loss."""
        x = torch.cat([encoding, action], dim=-1)
        return self.dynamics_net(x)


class RecoverFormer(nn.Module):
    """Full RecoverFormer policy network."""

    def __init__(
        self,
        obs_dim: int = 106,
        action_dim: int = 29,
        embed_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_modes: int = 4,
        mode_embed_dim: int = 32,
        n_contact_regions: int = 8,
        history_len: int = 50,
        adapt_hidden_dim: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.n_modes = n_modes

        # Components — dropout=0 for RL (stochastic forward breaks PPO importance sampling)
        self.encoder = CausalTransformerEncoder(obs_dim, embed_dim, n_heads, n_layers, dropout=0.0)
        self.mode_head = LatentRecoveryModeHead(embed_dim, n_modes, mode_embed_dim)
        self.affordance_head = ContactAffordanceHead(embed_dim, n_contact_regions)
        self.action_decoder = ActionDecoder(embed_dim, mode_embed_dim, n_contact_regions, action_dim)
        self.adaptation = AdaptationModule(embed_dim, action_dim, adapt_hidden_dim)

        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Temperature for Gumbel-Softmax (annealed during training)
        self.register_buffer("temperature", torch.tensor(1.0))

        # History buffer (managed externally during rollouts)
        self._history_buffer = None

    def forward(
        self,
        obs_history: torch.Tensor,
        use_adaptation: bool = False,
    ) -> dict:
        """
        Full forward pass.

        Args:
            obs_history: (batch, seq_len, obs_dim) or (batch, obs_dim) for single-step
            use_adaptation: whether to apply the adaptation residual
        Returns:
            dict with keys: action, value, mode_probs, affordances, encoding
        """
        # Handle single observation (wrap in history dim)
        if obs_history.dim() == 2:
            obs_history = obs_history.unsqueeze(1)

        # Encode
        encoding = self.encoder(obs_history)  # (B, embed_dim)

        # Mode prediction
        mode_embed, mode_probs = self.mode_head(
            encoding, temperature=self.temperature.item()
        )

        # Contact affordance prediction
        affordances = self.affordance_head(encoding)

        # Action decoding
        action = self.action_decoder(encoding, mode_embed, affordances)

        # Optional adaptation residual
        if use_adaptation:
            residual = self.adaptation(encoding)
            action = torch.clamp(action + residual, -1.0, 1.0)

        # Value estimate
        value = self.value_head(encoding)

        return {
            "action": action,
            "value": value.squeeze(-1),
            "mode_probs": mode_probs,
            "affordances": affordances,
            "encoding": encoding,
        }

    def get_action(self, obs_history: torch.Tensor, deterministic: bool = True,
                   use_adaptation: bool = False) -> np.ndarray:
        """Get action for deployment (numpy output)."""
        with torch.no_grad():
            out = self.forward(obs_history, use_adaptation=use_adaptation)
        return out["action"].cpu().numpy()

    def compute_adaptation_loss(self, encoding: torch.Tensor,
                                 action: torch.Tensor,
                                 next_encoding: torch.Tensor) -> torch.Tensor:
        """Compute self-supervised adaptation loss."""
        predicted = self.adaptation.predict_next_encoding(encoding, action)
        return F.mse_loss(predicted, next_encoding.detach())

    def compute_mode_loss(self, mode_probs: torch.Tensor) -> torch.Tensor:
        """Compute mode specialization loss.
        Encourages low entropy (decisive mode selection).
        """
        # Entropy of mode distribution (encourage low entropy = decisive)
        entropy = -(mode_probs * (mode_probs + 1e-8).log()).sum(dim=-1).mean()
        # We want LOW entropy, so loss = entropy
        return entropy

    def anneal_temperature(self, progress: float):
        """Anneal Gumbel-Softmax temperature. progress in [0, 1]."""
        # Anneal from 1.0 to 0.1
        new_temp = max(0.1, 1.0 - 0.9 * progress)
        self.temperature.fill_(new_temp)
