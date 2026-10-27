# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research implementation of the world-model architecture from LeCun's *A Path Towards Autonomous Machine Intelligence* (2022), applied to one task: an agent that writes tiny Python functions by planning in latent space. See `README.md` for the module-by-module mapping to the paper and `ROADMAP.md` for the longer "see, remember, reason" vision.

This repo is **public**. Never commit business figures, credentials, or anything from private Lær Smart repos.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch + numpy
.venv/bin/pip install pytest                                         # tests use pytest-style functions
.venv/bin/python -m pytest tests/                                    # sanity tests for the world and module contracts
.venv/bin/python -m alemayhu.train                                   # random play -> JEPA world model -> critic -> policy (a few minutes on CPU)
.venv/bin/python demo.py                                             # evaluation suite, verified on held-out inputs
.venv/bin/python demo.py --spec "a * b + a"                          # custom spec
.venv/bin/python demo.py --mode 1                                    # distilled reactive policy instead of planning
```

`demo.py` loads `checkpoints/agent.pt`, written by `alemayhu.train`. `checkpoints/` and `.venv` are gitignored.

## Layout

- `alemayhu/env/code_world.py` — `CodeWorld`: line-by-line edits from 17 primitive statements; executes on 8 probe inputs after each edit.
- `alemayhu/modules/` — one file per paper module: `perception`, `world_model` (JEPA predictor, VICReg), `cost` (immutable intrinsic + trainable critic), `memory`, `actor` (beam-search MPC + distilled policy), `configurator`.
- `alemayhu/agent.py` wires the modules; `alemayhu/train.py` runs the training phases.

## Invariants (keep faithful to the paper)

- The world model predicts **representations**, never raw percepts (JEPA, not generative).
- Intrinsic cost is **non-trainable** and must work on imagined states.
- **No code is executed during planning**; only the final program runs, against held-out tests the agent never perceives.
