"""Configurator module (paper section 3): executive control.

Given a task, it configures the other modules for the task at hand. Here
the task is "write a function whose behaviour matches this spec": the
configurator turns the spec into a goal embedding (the behaviour the
program should exhibit on the world's probe inputs) and primes the cost
module and the actor with it. One world-model engine, dynamically
configured per task — rather than a model per task.
"""

from typing import Callable

import torch

from ..env.code_world import goal_from_spec


class Configurator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.goal: torch.Tensor = None

    def configure_for_task(self, spec: Callable[[float, float], float]) -> torch.Tensor:
        """spec: the desired function. Returns the goal handed to the
        cost module and the actor."""
        g = goal_from_spec(spec)
        self.goal = torch.from_numpy(g).unsqueeze(0).to(self.device)
        return self.goal
