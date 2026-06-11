"""CodeWorld: the external world the agent acts in.

The world is a tiny Python programming session. The agent edits a program
line by line, drawn from a fixed pool of primitive statements over inputs
``a``, ``b`` and an accumulator ``r``. After every edit the world executes
the program on a fixed set of probe inputs and the resulting trace is part
of the next percept.

The agent never sees the task's hidden tests. It only perceives:
  - the program text (as a grid of line indices)
  - the execution trace on the probe inputs (the observable consequence
    of its actions in the world)

This makes the world dynamics non-trivial: to plan, the agent's world
model has to learn what each edit *does* to the semantics of the program,
purely from observing transitions.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

# Pool of primitive statements. `a`, `b` are the function arguments,
# `r` is the accumulator / return value (initialised to 0).
PRIMITIVES: List[str] = [
    "r = a",
    "r = b",
    "r = a + b",
    "r = a - b",
    "r = a * b",
    "r = r + a",
    "r = r + b",
    "r = r * a",
    "r = r * b",
    "r = r + 1",
    "r = r - 1",
    "r = r * 2",
    "r = -r",
    "r = r * r",
    "r = max(a, b)",
    "r = min(a, b)",
    "r = abs(r)",
]

MAX_LINES = 4

# Actions: append primitive k (0..K-1), delete last line, no-op (stop).
NUM_PRIMITIVES = len(PRIMITIVES)
ACTION_DELETE = NUM_PRIMITIVES
ACTION_NOOP = NUM_PRIMITIVES + 1
NUM_ACTIONS = NUM_PRIMITIVES + 2
ACTION_NAMES = PRIMITIVES + ["<delete>", "<noop>"]

# Probe inputs: the world executes the program on these after every edit
# and the outputs are part of the percept. Chosen to disambiguate the
# primitives (sign, asymmetry, magnitude).
PROBES: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (1.0, 0.0),
    (2.0, 3.0),
    (3.0, 2.0),
    (-1.0, 2.0),
    (2.0, -2.0),
    (-3.0, -1.0),
    (4.0, 5.0),
]
NUM_PROBES = len(PROBES)

# Outputs are squashed for the percept so the representation stays bounded.
OUTPUT_SCALE = 10.0


def run_program(lines: List[str], a: float, b: float) -> Optional[float]:
    """Execute a program on one input pair. Returns None on runtime error."""
    r = 0.0
    env = {"a": a, "b": b, "r": r, "max": max, "min": min, "abs": abs}
    try:
        for line in lines:
            exec(line, {"__builtins__": {}}, env)  # noqa: S102 - sandboxed DSL
        out = float(env["r"])
        if not np.isfinite(out):
            return None
        return out
    except Exception:
        return None


def trace(lines: List[str]) -> Tuple[np.ndarray, bool]:
    """Run program on all probes. Returns (squashed outputs, error flag)."""
    outs = np.zeros(NUM_PROBES, dtype=np.float32)
    error = False
    for i, (a, b) in enumerate(PROBES):
        out = run_program(lines, a, b)
        if out is None:
            error = True
            out = 0.0
        outs[i] = np.tanh(out / OUTPUT_SCALE)
    return outs, error


def goal_from_spec(fn: Callable[[float, float], float]) -> np.ndarray:
    """Encode a task spec (a target function) as the squashed trace it
    should produce on the probe inputs. This is what the configurator
    hands to the cost module as the goal."""
    g = np.array([fn(a, b) for a, b in PROBES], dtype=np.float32)
    return np.tanh(g / OUTPUT_SCALE)


@dataclass
class Observation:
    """A percept: program text grid + observable execution trace."""

    line_ids: List[int]          # indices into PRIMITIVES, len <= MAX_LINES
    outputs: np.ndarray          # squashed trace on probes, shape (NUM_PROBES,)
    error: bool                  # runtime error flag
    length: int                  # number of lines

    def to_vector(self) -> np.ndarray:
        """Flat percept vector fed to the perception module.

        Layout: [program one-hot grid | outputs | error | length/MAX_LINES]
        """
        grid = np.zeros((MAX_LINES, NUM_PRIMITIVES), dtype=np.float32)
        for row, lid in enumerate(self.line_ids):
            grid[row, lid] = 1.0
        return np.concatenate([
            grid.reshape(-1),
            self.outputs,
            np.array([1.0 if self.error else 0.0], dtype=np.float32),
            np.array([self.length / MAX_LINES], dtype=np.float32),
        ])


OBS_DIM = MAX_LINES * NUM_PRIMITIVES + NUM_PROBES + 2
# Index of the observable-consequence block inside the percept vector.
PERCEPT_FEATURES_START = MAX_LINES * NUM_PRIMITIVES
PERCEPT_FEATURES_DIM = NUM_PROBES + 2  # outputs + error + length


class CodeWorld:
    """The external, non-differentiable world (paper fig. 2, the globe)."""

    def __init__(self) -> None:
        self.lines: List[int] = []

    def reset(self) -> Observation:
        self.lines = []
        return self.observe()

    def observe(self) -> Observation:
        program = [PRIMITIVES[i] for i in self.lines]
        outputs, error = trace(program)
        return Observation(
            line_ids=list(self.lines),
            outputs=outputs,
            error=error,
            length=len(self.lines),
        )

    def step(self, action: int) -> Observation:
        if action < NUM_PRIMITIVES:
            if len(self.lines) < MAX_LINES:
                self.lines.append(action)
        elif action == ACTION_DELETE:
            if self.lines:
                self.lines.pop()
        # ACTION_NOOP: world unchanged
        return self.observe()

    def program_text(self) -> str:
        body = "\n".join(f"    {PRIMITIVES[i]}" for i in self.lines) or "    pass"
        return f"def f(a, b):\n    r = 0\n{body}\n    return r"

    def program_fn(self) -> Callable[[float, float], Optional[float]]:
        program = [PRIMITIVES[i] for i in self.lines]
        return lambda a, b: run_program(program, a, b)
