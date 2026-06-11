"""Short-term memory module (paper section 3).

Stores tuples of (percept, action, next percept, goal, intrinsic energy)
from past episodes. The world model trains its predictor from the stored
transitions; the critic trains by retrieving past states and the
intrinsic costs that followed them. A simple ring buffer plays the role
the paper assigns to a key-value associative memory.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray          # percept vector x[t]
    action: int              # a[t]
    next_obs: np.ndarray     # percept vector x[t+1]
    goal: np.ndarray         # goal active during the episode
    energy: float            # intrinsic energy of x[t+1]
    cost_to_go: float = 0.0  # discounted future intrinsic energy (filled at episode end)


class ShortTermMemory:
    def __init__(self, capacity: int = 100_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.pairs: Deque[tuple] = deque(maxlen=capacity)

    def store_episode(self, transitions: List[Transition], discount: float = 0.9) -> None:
        """Store an episode, back-filling observed cost-to-go for the critic."""
        ctg = 0.0
        for tr in reversed(transitions):
            ctg = tr.energy + discount * ctg
            tr.cost_to_go = ctg
        self.buffer.extend(transitions)
        # Keep consecutive pairs for multi-step world-model training.
        for first, second in zip(transitions, transitions[1:]):
            self.pairs.append((first, second))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_pairs(self, batch_size: int) -> List[tuple]:
        """Sample consecutive transition pairs (t, t+1) from one episode."""
        return random.sample(self.pairs, min(batch_size, len(self.pairs)))

    def __len__(self) -> int:
        return len(self.buffer)
