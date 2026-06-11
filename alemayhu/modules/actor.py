"""Actor module (paper sections 3, 3.1.1-3.1.3).

Two modes, after Kahneman:

Mode-1 (reactive): a policy network a = A(s) produces an action in a
single pass. It is trained by distilling the actions found by Mode-2
reasoning (paper fig. 5) — amortized inference.

Mode-2 (reasoning/planning): the actor proposes action sequences, the
world model *imagines* the resulting state sequence s[t+1] = Pred(s[t],
a[t]), the cost module scores each imagined state, and the actor keeps
the sequence with the lowest total energy

    F = stage_weight * sum_t C(s[t]) + C(s[T]) + critic(s[T])

then emits its first action (receding-horizon MPC). Stage costs are
down-weighted relative to the final state: reaching a good program may
require passing through bad intermediate ones (e.g. building `a + b`
before negating it), and a fully greedy sum prunes those paths. This is
the configurator modulating the cost module for the task at hand. The action space is
discrete, so the search is combinatorial — the paper explicitly allows
dynamic programming / heuristic search in this case; we use beam search.
No program is ever executed during planning: the search runs entirely in
representation space.
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from ..env.code_world import NUM_ACTIONS, NUM_PROBES
from .perception import STATE_DIM
from .world_model import Z_DIM


class Policy(nn.Module):
    """Mode-1 reactive policy A(s), conditioned on the configured goal."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + NUM_PROBES, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, goal], dim=-1))


@dataclass
class Plan:
    actions: List[int]
    energy: float


class Actor:
    def __init__(self, world_model, intrinsic_cost, critic, policy: Policy):
        self.world_model = world_model
        self.intrinsic_cost = intrinsic_cost
        self.critic = critic
        self.policy = policy

    @torch.no_grad()
    def act_mode1(self, s: torch.Tensor, goal: torch.Tensor) -> int:
        """Single-pass reactive action."""
        logits = self.policy(s, goal)
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def plan_mode2(
        self,
        s0: torch.Tensor,
        goal: torch.Tensor,
        horizon: int = 5,
        beam_width: int = 96,
        critic_weight: float = 1.0,
        stage_weight: float = 0.05,
        z_samples: int = 0,
        risk_weight: float = 0.5,
    ) -> Plan:
        """Beam search through the world model in representation space.

        s0: (1, STATE_DIM) current state estimate from perception.
        goal: (1, NUM_PROBES) goal configured by the configurator.
        Returns the minimum-energy action sequence found.

        Uncertainty (section 4.8): with z_samples > 0, each candidate
        transition is imagined under several samples of the latent z and
        scored as mean + risk_weight * std across samples, so the planner
        avoids action sequences whose outcomes the world model is unsure
        about. The rollout continues from the z = 0 (mode) prediction.
        Off by default: in this deterministic world it adds plan noise on
        near-tie decisions without improving robustness.
        """
        device = s0.device
        # Beam entries: state (B, STATE_DIM), accumulated energy (B,), actions
        states = s0
        energies = torch.zeros(1, device=device)
        histories: List[List[int]] = [[]]

        for _ in range(horizon):
            n = states.shape[0]
            # Expand every beam entry with every action.
            rep_states = states.repeat_interleave(NUM_ACTIONS, dim=0)
            actions = torch.arange(NUM_ACTIONS, device=device).repeat(n)
            next_states = self.world_model(rep_states, actions)
            g = goal.expand(next_states.shape[0], -1)
            step_cost = self.intrinsic_cost(next_states, g)

            if z_samples > 0:
                # Imagine the same transitions under sampled latents and
                # penalize candidates whose predicted cost is uncertain.
                m = rep_states.shape[0]
                zs = 0.5 * torch.randn(z_samples * m, Z_DIM, device=device)
                rep_k = rep_states.repeat(z_samples, 1)
                act_k = actions.repeat(z_samples)
                cost_k = self.intrinsic_cost(
                    self.world_model(rep_k, act_k, zs), g.repeat(z_samples, 1)
                ).view(z_samples, m)
                all_costs = torch.cat([step_cost.unsqueeze(0), cost_k], dim=0)
                step_cost = all_costs.mean(dim=0) + risk_weight * all_costs.std(dim=0)

            total = energies.repeat_interleave(NUM_ACTIONS) + stage_weight * step_cost

            # Prune with the critic as a value-to-go heuristic: a path
            # through a high-cost intermediate state survives if the
            # critic says good states are reachable from it.
            score = total + critic_weight * self.critic(next_states, g)
            keep = torch.topk(-score, k=min(beam_width, score.shape[0])).indices
            states = next_states[keep]
            energies = total[keep]
            histories = [
                histories[i // NUM_ACTIONS] + [int(i % NUM_ACTIONS)]
                for i in keep.tolist()
            ]

        # Final-state energy plus terminal cost-to-go from the critic.
        g = goal.expand(states.shape[0], -1)
        final = (
            energies
            + self.intrinsic_cost(states, g)
            + critic_weight * self.critic(states, g)
        )
        best = int(final.argmin().item())
        return Plan(actions=histories[best], energy=float(final[best].item()))
