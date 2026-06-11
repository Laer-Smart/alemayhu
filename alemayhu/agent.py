"""The full agent: wires the modules into perception-action loops
(paper sections 3.1.1 and 3.1.2)."""

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from .env.code_world import CodeWorld, ACTION_NOOP
from .modules import (
    Actor,
    Configurator,
    Critic,
    IntrinsicCost,
    Perception,
    ShortTermMemory,
    WorldModel,
)
from .modules.actor import Policy


@dataclass
class EpisodeResult:
    program: str
    fn: Callable[[float, float], Optional[float]]
    actions: List[int]
    final_energy: float


class Agent:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.perception = Perception().to(device)
        self.world_model = WorldModel().to(device)
        self.intrinsic_cost = IntrinsicCost()
        self.critic = Critic().to(device)
        self.policy = Policy().to(device)
        self.memory = ShortTermMemory()
        self.configurator = Configurator(device)
        self.actor = Actor(
            self.world_model, self.intrinsic_cost, self.critic, self.policy
        )

    def encode(self, obs) -> torch.Tensor:
        x = torch.from_numpy(obs.to_vector()).unsqueeze(0).to(self.device)
        return self.perception(x)

    @torch.no_grad()
    def write_code(
        self,
        spec: Callable[[float, float], float],
        mode: int = 2,
        max_steps: int = 6,
        horizon: int = 4,
        beam_width: int = 64,
        verbose: bool = False,
    ) -> EpisodeResult:
        """Perception-action episode: the agent writes a program whose
        behaviour matches `spec`.

        mode=2: receding-horizon planning through the world model.
        mode=1: reactive policy only (distilled from Mode-2).
        """
        goal = self.configurator.configure_for_task(spec)
        world = CodeWorld()
        obs = world.reset()
        actions: List[int] = []

        for step in range(max_steps):
            s = self.encode(obs)
            if mode == 2:
                plan = self.actor.plan_mode2(
                    s, goal, horizon=min(horizon, max_steps - step),
                    beam_width=beam_width,
                )
                action = plan.actions[0]
            else:
                action = self.actor.act_mode1(s, goal)

            if verbose:
                from .env.code_world import ACTION_NAMES
                print(f"  step {step}: {ACTION_NAMES[action]}")

            obs = world.step(action)
            actions.append(action)

            # Settled on a program: agent chooses to stop editing.
            if action == ACTION_NOOP:
                break

        s = self.encode(obs)
        energy = float(self.intrinsic_cost(s, goal).item())
        return EpisodeResult(
            program=world.program_text(),
            fn=world.program_fn(),
            actions=actions,
            final_energy=energy,
        )
