"""Sanity tests for the world, the modules, and their contracts."""

import numpy as np
import torch

from alemayhu.env.code_world import (
    ACTION_DELETE,
    ACTION_NOOP,
    NUM_ACTIONS,
    OBS_DIM,
    CodeWorld,
    goal_from_spec,
    run_program,
)
from alemayhu.modules import IntrinsicCost, Perception, WorldModel
from alemayhu.modules.perception import STATE_DIM
from alemayhu.modules.world_model import Z_DIM, vicreg_regularizer


def test_world_executes_edits():
    world = CodeWorld()
    obs = world.reset()
    assert obs.length == 0
    obs = world.step(2)  # "r = a + b"
    assert obs.length == 1
    assert not obs.error
    fn = world.program_fn()
    assert fn(2.0, 3.0) == 5.0


def test_delete_and_noop():
    world = CodeWorld()
    world.reset()
    world.step(0)
    obs = world.step(ACTION_DELETE)
    assert obs.length == 0
    obs = world.step(ACTION_NOOP)
    assert obs.length == 0


def test_observation_vector_shape():
    world = CodeWorld()
    obs = world.reset()
    assert obs.to_vector().shape == (OBS_DIM,)


def test_run_program_error_returns_none():
    assert run_program(["r = r / 0"], 1.0, 1.0) is None


def test_perception_state_layout():
    p = Perception()
    world = CodeWorld()
    obs = world.reset()
    x = torch.from_numpy(obs.to_vector()).unsqueeze(0)
    s = p(x)
    assert s.shape == (1, STATE_DIM)
    # Percept-feature block is copied verbatim (hard-wired view for the
    # immutable intrinsic cost).
    np.testing.assert_allclose(s[0, :8].detach().numpy(), obs.outputs, atol=1e-6)


def test_world_model_shapes():
    wm = WorldModel()
    s = torch.randn(4, STATE_DIM)
    a = torch.randint(0, NUM_ACTIONS, (4,))
    z = torch.zeros(4, Z_DIM)
    assert wm(s, a, z).shape == (4, STATE_DIM)


def test_latent_inference_reduces_energy():
    wm = WorldModel()
    s = torch.randn(8, STATE_DIM)
    a = torch.randint(0, NUM_ACTIONS, (8,))
    sy = torch.randn(8, STATE_DIM)
    e0 = ((sy - wm(s, a)) ** 2).sum().item()
    z = wm.infer_latent(s, a, sy, steps=8)
    e1 = ((sy - wm(s, a, z)) ** 2).sum().item()
    assert e1 <= e0 + 1e-5


def test_intrinsic_cost_zero_at_goal():
    cost = IntrinsicCost()
    spec = lambda a, b: a + b  # noqa: E731
    goal = torch.from_numpy(goal_from_spec(spec)).unsqueeze(0)
    world = CodeWorld()
    world.reset()
    obs = world.step(2)  # "r = a + b"
    s = torch.zeros(1, STATE_DIM)
    s[0, :8] = torch.from_numpy(obs.outputs)
    s[0, 8] = 0.0  # no error
    s[0, 9] = obs.length / 4
    energy = cost(s, goal).item()
    assert energy < 0.01  # only the length pressure remains


def test_vicreg_penalizes_collapse():
    collapsed = torch.zeros(32, 16)
    spread = torch.randn(32, 16)
    assert vicreg_regularizer(collapsed) > vicreg_regularizer(spread)
