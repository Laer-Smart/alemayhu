# 🪬 alemayhu

System to see, remember, and reason about the world.

An implementation of the world-model architecture from Yann LeCun's
position paper [*A Path Towards Autonomous Machine Intelligence*
(2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf), applied to a concrete
task: **an agent that writes code by planning in latent space**.

## The architecture

Every module from the paper's figure 2 is implemented:

| Module | Paper role | Here |
|---|---|---|
| **Perception** | `s = Enc(x)` — estimate the state of the world | encodes the program + its observable execution trace ([`modules/perception.py`](alemayhu/modules/perception.py)) |
| **World model** | `s[t+1] = Pred(s[t], a[t], z[t])` — predict future world states in representation space | a JEPA predictor that learns what each code edit *does*, without executing code ([`modules/world_model.py`](alemayhu/modules/world_model.py)) |
| **Cost** | scalar energy = immutable intrinsic cost + trainable critic | hard-wired discomfort (behaviour ≠ goal, runtime errors, bloat) + learned cost-to-go ([`modules/cost.py`](alemayhu/modules/cost.py)) |
| **Short-term memory** | store states, actions and energies; trains world model and critic | episodic transition buffer ([`modules/memory.py`](alemayhu/modules/memory.py)) |
| **Actor** | Mode-1 reactive policy, Mode-2 planning by energy minimization | beam-search MPC through the world model + a policy distilled from it ([`modules/actor.py`](alemayhu/modules/actor.py)) |
| **Configurator** | configure the other modules for the task at hand | turns a spec into a goal that primes the cost module and actor ([`modules/configurator.py`](alemayhu/modules/configurator.py)) |

Key properties kept faithful to the paper:

- **JEPA, not generative.** The world model predicts the *representation*
  of the next state, never the raw percept. Trained with the VICReg
  criterion (variance hinge + covariance decorrelation through an
  expander, §4.5.1) so the representation doesn't collapse.
- **Latent variable `z`.** Low-dimensional, regularized with `R(z) = ‖z‖²`,
  inferred by gradient descent on the energy during training (§4.4, §4.8).
- **Self-supervised.** The world model is trained purely from observed
  transitions under random play — no rewards, no labels.
- **Immutable intrinsic cost.** Non-trainable, and applicable to
  *imagined* states predicted by the world model, which is what makes
  planning possible (§3).
- **Mode-2 reasoning as energy minimization.** Planning = search for the
  action sequence minimizing `F = Σ C(s[t]) + critic(s[T])` through the
  world model — receding-horizon MPC (§3.1.2). The action space is
  discrete, so the search is combinatorial (beam search), as the paper
  prescribes for discretized actions.
- **Mode-2 → Mode-1 distillation.** The reactive policy is trained to
  imitate planned actions — amortized inference, "compiling" reasoning
  into a skill (§3.1.3, fig. 5).

## The world

`CodeWorld` ([`env/code_world.py`](alemayhu/env/code_world.py)): the agent
edits a tiny Python function line-by-line from a pool of 17 primitive
statements. After every edit the world executes the program on 8 probe
inputs; the resulting trace is part of the next percept. The agent's
world model must learn the *semantics of code* — what a given edit will
do to the program's behaviour — purely from watching transitions.

At planning time **no code is executed**: the agent imagines edit
sequences entirely in representation space, picks the lowest-energy one,
and only the final written program is run (against held-out tests it
never perceives).

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# train: random play -> JEPA world model -> critic -> policy distillation
.venv/bin/python -m alemayhu.train

# the agent writes code for an evaluation suite, verified on held-out tests
.venv/bin/python demo.py

# write code for your own spec
.venv/bin/python demo.py --spec "a * b + a"

# compare deliberate planning vs the distilled reactive policy
.venv/bin/python demo.py --mode 1
```

Training takes a few minutes on CPU.

## Results

On a 12-spec evaluation suite, verified against held-out test inputs the
agent never observes (seed 0):

- **Mode-2 (planning through the world model): 10/12.** The two failures
  (`-(a + b)`, occasionally `a * a`) are cases where the search finds an
  action sequence whose *imagined* outcome the world model scores
  slightly too optimistically — the classic failure mode of planning
  through a learned model. Results vary a little run to run.
- **Mode-1 (distilled reactive policy): 3/12.** A single forward pass
  through a small MLP solves the one-line specs; multi-step compositions
  still need deliberate Mode-2 reasoning. This gap is the paper's point:
  Mode-2 is onerous but general, Mode-1 is cheap but only covers
  practiced skills.

Things that mattered, found the hard way:

- The critic must be retrained on the agent's *competent* behaviour.
  Trained only on random play, it estimates "future discomfort under
  random actions" — which is high everywhere and poisons planning.
  Training alternates Mode-2 practice rounds with critic refits.
- Stage costs in the planning energy must be down-weighted relative to
  the final state, or the search prunes every path that passes through a
  bad intermediate state (you cannot build `a + b` before negating it if
  intermediate discomfort is fully priced).
- Multi-step rollout consistency (predict two steps ahead through the
  predictor and match the encoder) substantially reduces the compounding
  model error that planning otherwise exploits. It also nearly doubled
  Mode-2 → Mode-1 imitation accuracy (0.42 → 0.80).

## Not yet implemented (see [ROADMAP.md](ROADMAP.md))

- **H-JEPA** (§4.6): stacked JEPAs predicting at multiple time scales.
- **Hierarchical planning** (§4.7): high-level latent "actions" as
  subgoal conditions for a lower level.
- Configurator that *learns* to modulate module parameters, rather than
  only supplying the goal.
- Uncertainty-aware planning: sampling `z` during rollout to plan against
  multiple plausible futures (§4.8).
