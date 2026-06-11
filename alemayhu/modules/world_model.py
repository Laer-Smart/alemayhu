"""World model module (paper sections 3, 4.4-4.5): s[t+1] = Pred(s[t], a[t], z[t]).

A JEPA predictor: it predicts the *representation* of the next world
state, not the raw next percept. Trained self-supervised with the VICReg
criterion (paper fig. 14):

  - prediction error  D(sy, ~sy) = || sy - Pred(sx, a, z) ||^2
  - variance hinge    keeps every component's std above a threshold
  - covariance        decorrelates components
  (both applied to expander embeddings of sx and sy)

The latent variable z carries the information about s[t+1] that is not
predictable from (s[t], a[t]). Following section 4.8.1 it is kept
low-dimensional and regularized (R(z) = ||z||^2) to limit its information
content and prevent energy collapse. During training z-bar is inferred by
gradient descent on the energy; during planning the agent uses z = 0 (the
mode of the regularizer's Gibbs distribution) and may sample z to probe
uncertainty.
"""

import torch
import torch.nn as nn

from ..env.code_world import NUM_ACTIONS
from .perception import STATE_DIM

Z_DIM = 2


class WorldModel(nn.Module):
    def __init__(self, hidden: int = 192):
        super().__init__()
        self.action_embed = nn.Embedding(NUM_ACTIONS, 24)
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 24 + Z_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(
        self, s: torch.Tensor, a: torch.Tensor, z: torch.Tensor = None
    ) -> torch.Tensor:
        """s: (B, STATE_DIM), a: (B,) long, z: (B, Z_DIM) -> (B, STATE_DIM).

        Residual prediction: most edits change the state locally, so the
        predictor models the change.
        """
        if z is None:
            z = torch.zeros(s.shape[0], Z_DIM, device=s.device)
        inp = torch.cat([s, self.action_embed(a), z], dim=-1)
        return s + self.net(inp)

    def infer_latent(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        sy: torch.Tensor,
        steps: int = 4,
        lr: float = 0.1,
        reg: float = 1.0,
    ) -> torch.Tensor:
        """Infer z-bar = argmin_z ||sy - Pred(s, a, z)||^2 + reg * ||z||^2.

        Gradient-based latent inference (paper section 4.4). The model
        parameters are frozen during this inner loop.
        """
        z = torch.zeros(s.shape[0], Z_DIM, device=s.device, requires_grad=True)
        for _ in range(steps):
            pred = self.forward(s, a, z)
            energy = ((sy - pred) ** 2).sum(dim=-1).sum() + reg * (z ** 2).sum()
            (grad,) = torch.autograd.grad(energy, z, create_graph=False)
            z = (z - lr * grad).detach().requires_grad_(True)
        return z.detach()


class Expander(nn.Module):
    """VICReg expander: maps representations to a higher-dimensional space
    where the variance/covariance criteria are applied (paper fig. 14)."""

    def __init__(self, dim: int = STATE_DIM, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out),
            nn.GELU(),
            nn.Linear(out, out),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def vicreg_regularizer(v: torch.Tensor, var_target: float = 1.0) -> torch.Tensor:
    """Variance hinge + covariance loss over a batch of embeddings v: (B, D)."""
    v = v - v.mean(dim=0)
    std = torch.sqrt(v.var(dim=0) + 1e-4)
    var_loss = torch.relu(var_target - std).mean()
    n, d = v.shape
    cov = (v.T @ v) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss + 0.04 * cov_loss
