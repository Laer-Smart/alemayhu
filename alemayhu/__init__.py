"""Alemayhu — an implementation of Yann LeCun's world-model architecture.

Reference: LeCun, "A Path Towards Autonomous Machine Intelligence" (2022).

Modules (paper section 3):
  - perception     Enc(x): estimates the state of the world
  - world_model    Pred(s, a, z): JEPA predictor, trained self-supervised
  - cost           intrinsic cost (immutable) + critic (trainable)
  - memory         short-term memory of states, actions and energies
  - actor          Mode-1 reactive policy + Mode-2 planning (MPC)
  - configurator   configures cost/actor for the task at hand
"""

__version__ = "0.1.0"
