"""Cost module (paper section 3): energy = intrinsic cost + critic.

Intrinsic Cost: immutable (non-trainable). Computes the instantaneous
"discomfort" of a state. It reads only the hard-wired percept-feature
block of the state vector (execution trace, error flag, program length),
so it applies equally to states encoded by perception and to states
imagined by the world model — exactly the property the paper requires for
planning. The configurator modulates it with the goal of the task at
hand.

Trainable Critic: predicts future intrinsic energy (cost-to-go) from a
state. Trained by retrieving (state, observed cost-to-go) pairs from the
short-term memory, as described in the paper.
"""

import torch
import torch.nn as nn

from ..env.code_world import NUM_PROBES
from .perception import STATE_DIM

# Layout of the hard-wired feature block at the front of s.
OUT_SLICE = slice(0, NUM_PROBES)
ERR_IDX = NUM_PROBES
LEN_IDX = NUM_PROBES + 1

ERROR_WEIGHT = 1.0
LENGTH_WEIGHT = 0.005


class IntrinsicCost(nn.Module):
    """Immutable. No parameters; .forward is the hard-wired energy."""

    def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """s: (B, STATE_DIM), goal: (B, NUM_PROBES) -> energy (B,).

        Discomfort = distance between the program's observable behaviour
        and the goal behaviour, plus pain for runtime errors and a mild
        pressure toward short programs.
        """
        mismatch = ((s[:, OUT_SLICE] - goal) ** 2).mean(dim=-1)
        error_pain = ERROR_WEIGHT * s[:, ERR_IDX].clamp(0.0, 1.0)
        length_pressure = LENGTH_WEIGHT * s[:, LEN_IDX].clamp(0.0, 1.0)
        return mismatch + error_pain + length_pressure


class Critic(nn.Module):
    """Trainable. Predicts discounted future intrinsic energy."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + NUM_PROBES, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # energies are non-negative
        )

    def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, goal], dim=-1)).squeeze(-1)
