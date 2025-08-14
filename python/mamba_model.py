"""
Mamba Model Implementation for Trading

This module implements the Mamba architecture (Selective State Space Model)
optimized for financial time series prediction.

References:
    - Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling
      with Selective State Spaces." arXiv preprint arXiv:2312.00752.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MambaBlock(nn.Module):
    """
    Core Mamba block implementing selective state space model.

    The block consists of:
    1. Input projection with gating
    2. Causal convolution for local context
    3. Selective SSM for long-range dependencies
    4. Output projection

    Args:
        d_model: Model dimension
        d_state: SSM state dimension (N in paper)
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dt_rank: Rank for delta projection (default: auto)
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        dt_init: Delta initialization method ('random' or 'constant')
        dt_scale: Scale for delta initialization
        bias: Whether to use bias in linear layers
        conv_bias: Whether to use bias in convolution
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # Input projection: projects to 2*d_inner for x and z (gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        # SSM parameters projection
        # Projects to: dt_rank (for delta) + d_state (for B) + d_state (for C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # Delta projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize delta projection bias
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize delta bias to ensure dt is in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (diagonal, learned in log space for stability)
        # Initialized with S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection and split into x and z (gate)
        xz = self.in_proj(x)  # (batch, seq_len, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)

        # Causal convolution
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal: remove future padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)
        x = F.silu(x)

        # Selective SSM
        y = self.ssm(x)

        # Gating with SiLU activation
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def ssm(self, x: Tensor) -> Tensor:
        """
        Selective State Space Model computation.

        Implements the selective scan with input-dependent parameters:
            h_t = Ā_t * h_{t-1} + B̄_t * x_t
            y_t = C_t * h_t + D * x_t

        Args:
            x: Input after convolution, shape (batch, seq_len, d_inner)

        Returns:
            SSM output, shape (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # Project to get dt, B, C (all input-dependent)
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Compute delta (step size) with softplus for positivity
        dt = F.softplus(self.dt_proj(dt))  # (batch, seq_len, d_inner)

        # Get A from log space (negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize: Ā = exp(dt * A), B̄ = dt * B
        # For efficiency, we use the simplified discretization
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)

        # Selective scan (sequential for correctness, can be parallelized)
        h = torch.zeros(
            batch, d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        ys = []

        for t in range(seq_len):
            # State update: h_t = Ā_t * h_{t-1} + B̄_t * x_t
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)
            # Output: y_t = C_t * h_t
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (batch, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)

        # Add skip connection: y = y + D * x
        y = y + x * self.D

        return y


class MambaLayer(nn.Module):
    """
    Full Mamba layer with normalization and residual connection.

    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with pre-norm and residual."""
        return x + self.mamba(self.norm(x))


class MambaTradingModel(nn.Module):
    """
    Complete Mamba model for trading signal prediction.

    Architecture:
        1. Input projection (features -> d_model)
        2. Stack of Mamba layers
        3. Output head for classification/regression

    Args:
        n_features: Number of input features
        d_model: Model dimension
        n_layers: Number of Mamba layers
        d_state: SSM state dimension
        d_conv: Convolution kernel size
        expand: Expansion factor
        n_classes: Number of output classes (3 for buy/hold/sell)
        dropout: Dropout rate
        task: 'classification' or 'regression'
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_classes: int = 3,
        dropout: float = 0.1,
        task: str = "classification",
    ):
        super().__init__()
        self.task = task

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

        # Output head
        if task == "classification":
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes),
            )
        else:  # regression
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

    def forward(
        self,
        x: Tensor,
        return_all_steps: bool = False,
    ) -> Tensor:
        """
        Forward pass for trading prediction.

        Args:
            x: Input features of shape (batch, seq_len, n_features)
            return_all_steps: If True, return predictions for all timesteps

        Returns:
            If return_all_steps: (batch, seq_len, n_classes/1)
            Else: (batch, n_classes/1) for last timestep only
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_dropout(x)

        # Mamba layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        # Output
        if return_all_steps:
            return self.output_head(x)
        else:
            # Use only the last timestep for prediction
            return self.output_head(x[:, -1])

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Get prediction probabilities for classification.

        Args:
            x: Input features of shape (batch, seq_len, n_features)

        Returns:
            Probabilities of shape (batch, n_classes)
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")

        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def generate_signals(
        self,
        x: Tensor,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
    ) -> Tuple[list, Tensor]:
        """
        Generate trading signals from model predictions.

        Args:
            x: Input features of shape (batch, seq_len, n_features)
            buy_threshold: Minimum probability for buy signal
            sell_threshold: Minimum probability for sell signal

        Returns:
            Tuple of (signals list, probabilities tensor)
            signals: List of ('BUY', 'SELL', or 'HOLD', confidence)
        """
        probs = self.predict_proba(x)

        signals = []
        for prob in probs:
            # Assuming class order: [SELL, HOLD, BUY]
            sell_prob, hold_prob, buy_prob = prob.tolist()

            if buy_prob > buy_threshold:
                signals.append(('BUY', buy_prob))
            elif sell_prob > sell_threshold:
                signals.append(('SELL', sell_prob))
            else:
                signals.append(('HOLD', hold_prob))

        return signals, probs


def create_mamba_trading_model(
    n_features: int,
    preset: str = "default",
    **kwargs,
) -> MambaTradingModel:
    """
    Factory function to create Mamba trading models with presets.

    Args:
        n_features: Number of input features
        preset: Model preset ('small', 'default', 'large')
        **kwargs: Override preset parameters

    Returns:
        Configured MambaTradingModel
    """
    presets = {
        "small": {
            "d_model": 32,
            "n_layers": 2,
            "d_state": 8,
            "d_conv": 4,
            "expand": 2,
        },
        "default": {
            "d_model": 64,
            "n_layers": 4,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        },
        "large": {
            "d_model": 128,
            "n_layers": 6,
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
        },
    }

    config = presets.get(preset, presets["default"])
    config.update(kwargs)

    return MambaTradingModel(n_features=n_features, **config)
