"""Perception module (paper section 3): s = Enc(x).

Estimates the state of the world from a percept. The representation is
split in two blocks:

  s = [ percept features | learned embedding ]

The percept-feature block is copied straight from the observable
consequences in the percept (execution trace, error flag, program
length). Keeping these dimensions un-learned gives the immutable
intrinsic cost a fixed, hard-wired view of the state — the paper requires
the intrinsic cost to be non-trainable, and it must be applicable both to
encoded real states and to states *imagined* by the world model. Because
the JEPA predictor has to predict the full s vector, it is forced to
learn the semantics of code edits (what the trace will look like) without
ever executing a program.

The learned block is a small MLP embedding of the whole percept, trained
end-to-end with the JEPA criterion (VICReg keeps it from collapsing).
"""

import torch
import torch.nn as nn

from ..env.code_world import (
    OBS_DIM,
    PERCEPT_FEATURES_DIM,
    PERCEPT_FEATURES_START,
)

LEARNED_DIM = 48
STATE_DIM = PERCEPT_FEATURES_DIM + LEARNED_DIM
FEATURES_SLICE = slice(0, PERCEPT_FEATURES_DIM)


class Perception(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, LEARNED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, OBS_DIM) percept vectors -> s: (B, STATE_DIM)."""
        feats = x[:, PERCEPT_FEATURES_START:PERCEPT_FEATURES_START + PERCEPT_FEATURES_DIM]
        learned = self.mlp(x)
        return torch.cat([feats, learned], dim=-1)
