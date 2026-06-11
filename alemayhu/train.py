"""Training, in three phases mirroring the paper.

Phase 1 — self-supervised world-model learning (sections 4.4-4.5):
    The agent acts with random edits ("play"), observes transitions, and
    trains the JEPA: predict the representation of the next world state
    from the current state and the action. VICReg prevents collapse; the
    latent z absorbs whatever is not predictable. No rewards, no labels.

Phase 2 — critic training (section 3):
    The critic learns to predict observed discounted future intrinsic
    energy from (state, goal) pairs retrieved from short-term memory.

Phase 3 — Mode-2 to Mode-1 distillation (section 3.1.3, fig. 5):
    The agent plans with the world model (Mode-2) on practice tasks, and
    the reactive policy is trained to imitate the planned actions —
    amortized inference, "compiling" deliberate reasoning into a skill.
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .agent import Agent
from .env.code_world import (
    ACTION_NOOP,
    MAX_LINES,
    NUM_ACTIONS,
    NUM_PRIMITIVES,
    PRIMITIVES,
    CodeWorld,
    trace,
)
from .modules.cost import ERROR_WEIGHT, LENGTH_WEIGHT, Critic
from .modules.memory import ShortTermMemory
from .modules.perception import PERCEPT_FEATURES_DIM
from .modules.memory import Transition
from .modules.world_model import Expander, vicreg_regularizer

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "agent.pt"


def random_reachable_goal(rng: random.Random) -> np.ndarray:
    """Sample a practice goal: the behaviour of a random short program.
    Practicing against reachable goals is the agent's 'play'."""
    depth = rng.randint(1, MAX_LINES - 1)
    lines = [PRIMITIVES[rng.randrange(NUM_PRIMITIVES)] for _ in range(depth)]
    outputs, error = trace(lines)
    if error:
        return random_reachable_goal(rng)
    return outputs


def intrinsic_energy_np(obs, goal: np.ndarray) -> float:
    """Same hard-wired energy as IntrinsicCost, computed on a raw percept."""
    mismatch = float(((obs.outputs - goal) ** 2).mean())
    return (
        mismatch
        + ERROR_WEIGHT * (1.0 if obs.error else 0.0)
        + LENGTH_WEIGHT * obs.length / MAX_LINES
    )


def collect_play_episodes(agent: Agent, n_episodes: int, max_steps: int,
                          seed: int = 0) -> None:
    """Random-action exploration. Fills short-term memory with transitions."""
    rng = random.Random(seed)
    for _ in range(n_episodes):
        goal = random_reachable_goal(rng)
        world = CodeWorld()
        obs = world.reset()
        episode: List[Transition] = []
        for _ in range(max_steps):
            action = rng.randrange(NUM_ACTIONS)
            next_obs = world.step(action)
            episode.append(
                Transition(
                    obs=obs.to_vector(),
                    action=action,
                    next_obs=next_obs.to_vector(),
                    goal=goal.copy(),
                    energy=intrinsic_energy_np(next_obs, goal),
                )
            )
            obs = next_obs
        agent.memory.store_episode(episode)


def batches(memory, batch_size: int, device: str):
    sample = memory.sample(batch_size)
    x = torch.from_numpy(np.stack([t.obs for t in sample])).to(device)
    a = torch.tensor([t.action for t in sample], dtype=torch.long, device=device)
    xn = torch.from_numpy(np.stack([t.next_obs for t in sample])).to(device)
    g = torch.from_numpy(np.stack([t.goal for t in sample])).to(device)
    ctg = torch.tensor([t.cost_to_go for t in sample], dtype=torch.float32,
                       device=device)
    return x, a, xn, g, ctg


def train_world_model(agent: Agent, steps: int, batch_size: int = 256,
                      lr: float = 1e-3, log_every: int = 200) -> None:
    """Phase 1: JEPA with VICReg. Trains perception + predictor jointly."""
    expander = Expander().to(agent.device)
    params = (
        list(agent.perception.parameters())
        + list(agent.world_model.parameters())
        + list(expander.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr)

    for step in range(1, steps + 1):
        x, a, xn, _, _ = batches(agent.memory, batch_size, agent.device)
        sx = agent.perception(x)
        sy = agent.perception(xn)

        # Infer the latent that best explains the observed transition,
        # then take the parameter gradient at that latent.
        z = agent.world_model.infer_latent(sx.detach(), a, sy.detach(), steps=2)
        pred = agent.world_model(sx, a, z)

        # The percept-feature block carries the observable consequences the
        # intrinsic cost reads during planning — weight its accuracy higher
        # (section 4.5.2: biasing a JEPA towards task-relevant predictions).
        pf = PERCEPT_FEATURES_DIM
        pred_loss = F.mse_loss(pred, sy) + 4.0 * F.mse_loss(pred[:, :pf], sy[:, :pf])

        # Multi-step rollout consistency: applying the predictor twice
        # must match the encoding two steps ahead. Planning rolls the
        # predictor over several steps, so single-step accuracy is not
        # enough — compounding errors get exploited by the search.
        first, second = zip(*agent.memory.sample_pairs(batch_size))
        x0 = torch.from_numpy(np.stack([t.obs for t in first])).to(agent.device)
        a0 = torch.tensor([t.action for t in first], dtype=torch.long, device=agent.device)
        a1 = torch.tensor([t.action for t in second], dtype=torch.long, device=agent.device)
        x2 = torch.from_numpy(np.stack([t.next_obs for t in second])).to(agent.device)
        s2 = agent.perception(x2)
        roll = agent.world_model(agent.world_model(agent.perception(x0), a0), a1)
        roll_loss = F.mse_loss(roll, s2) + 4.0 * F.mse_loss(roll[:, :pf], s2[:, :pf])

        reg_loss = vicreg_regularizer(expander(sx)) + vicreg_regularizer(expander(sy))
        loss = pred_loss + roll_loss + reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0:
            print(f"  [world model] step {step}/{steps} "
                  f"pred {pred_loss.item():.5f} reg {reg_loss.item():.5f}")


def train_critic(agent: Agent, steps: int, batch_size: int = 256,
                 lr: float = 1e-3, log_every: int = 200) -> None:
    """Phase 2: predict observed cost-to-go from (s, goal)."""
    opt = torch.optim.Adam(agent.critic.parameters(), lr=lr)
    for step in range(1, steps + 1):
        x, _, _, g, ctg = batches(agent.memory, batch_size, agent.device)
        with torch.no_grad():
            s = agent.perception(x)
        loss = F.mse_loss(agent.critic(s, g), ctg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0:
            print(f"  [critic] step {step}/{steps} mse {loss.item():.5f}")


def run_mode2_practice(agent: Agent, n_episodes: int, memory,
                       max_steps: int = 6, seed: int = 1,
                       log_every: int = 50, tag: str = "practice"
                       ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Run Mode-2 planning episodes on practice goals. Stores the
    transitions (with observed energies) into `memory` so the critic can
    be retrained on competent behaviour, and returns the state-action
    pairs for policy distillation."""
    rng = random.Random(seed)
    dataset: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for ep in range(1, n_episodes + 1):
        goal_np = random_reachable_goal(rng)
        goal = torch.from_numpy(goal_np).unsqueeze(0).to(agent.device)
        world = CodeWorld()
        obs = world.reset()
        episode: List[Transition] = []
        for step in range(max_steps):
            s = agent.encode(obs)
            plan = agent.actor.plan_mode2(
                s, goal, horizon=min(4, max_steps - step), beam_width=48
            )
            action = plan.actions[0]
            dataset.append((obs.to_vector(), goal_np.copy(), action))
            next_obs = world.step(action)
            episode.append(
                Transition(
                    obs=obs.to_vector(),
                    action=action,
                    next_obs=next_obs.to_vector(),
                    goal=goal_np.copy(),
                    energy=intrinsic_energy_np(next_obs, goal_np),
                )
            )
            obs = next_obs
            if action == ACTION_NOOP:
                break
        memory.store_episode(episode)
        if ep % log_every == 0:
            print(f"  [{tag}] episode {ep}/{n_episodes} "
                  f"({len(dataset)} state-action pairs)")
    return dataset


def refit_critic(agent: Agent, memory, steps: int = 800,
                 batch_size: int = 256, lr: float = 1e-3) -> None:
    """Retrain the critic from scratch on a memory of competent (Mode-2)
    behaviour: its cost-to-go estimates must reflect how the agent will
    actually behave, not the random play of phase 0."""
    agent.critic = Critic().to(agent.device)
    agent.actor.critic = agent.critic
    opt = torch.optim.Adam(agent.critic.parameters(), lr=lr)
    for step in range(1, steps + 1):
        x, _, _, g, ctg = batches(memory, batch_size, agent.device)
        with torch.no_grad():
            s = agent.perception(x)
        loss = F.mse_loss(agent.critic(s, g), ctg)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  [critic refit] final mse {loss.item():.5f} "
          f"on {len(memory)} Mode-2 transitions")


def fit_policy(agent: Agent, dataset, lr: float = 1e-3) -> None:
    """Distill Mode-2 actions into the Mode-1 reactive policy (fig. 5)."""
    opt = torch.optim.Adam(agent.policy.parameters(), lr=lr)
    x = torch.from_numpy(np.stack([d[0] for d in dataset])).to(agent.device)
    g = torch.from_numpy(np.stack([d[1] for d in dataset])).to(agent.device)
    a = torch.tensor([d[2] for d in dataset], dtype=torch.long, device=agent.device)
    with torch.no_grad():
        s = agent.perception(x)
    for epoch in range(300):
        logits = agent.policy(s, g)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc = (logits.argmax(-1) == a).float().mean().item()
    print(f"  [distill] imitation accuracy {acc:.3f}")


def save(agent: Agent) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "perception": agent.perception.state_dict(),
            "world_model": agent.world_model.state_dict(),
            "critic": agent.critic.state_dict(),
            "policy": agent.policy.state_dict(),
        },
        CHECKPOINT,
    )
    print(f"saved checkpoint -> {CHECKPOINT}")


def load(agent: Agent) -> Agent:
    state = torch.load(CHECKPOINT, map_location=agent.device)
    agent.perception.load_state_dict(state["perception"])
    agent.world_model.load_state_dict(state["world_model"])
    agent.critic.load_state_dict(state["critic"])
    agent.policy.load_state_dict(state["policy"])
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the world-model agent")
    parser.add_argument("--episodes", type=int, default=4000,
                        help="random-play episodes for world-model learning")
    parser.add_argument("--wm-steps", type=int, default=3000)
    parser.add_argument("--critic-steps", type=int, default=1500)
    parser.add_argument("--distill-episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    agent = Agent()

    print("phase 0: collecting play episodes (random edits, no rewards)")
    collect_play_episodes(agent, args.episodes, max_steps=6, seed=args.seed)
    print(f"  {len(agent.memory)} transitions in short-term memory")

    print("phase 1: self-supervised world-model training (JEPA + VICReg)")
    train_world_model(agent, steps=args.wm_steps)

    print("phase 2: critic training (cost-to-go from memory)")
    train_critic(agent, steps=args.critic_steps)

    # Phase 3: alternate Mode-2 practice and critic refits. The critic
    # from phase 2 reflects random behaviour; each round replaces it with
    # cost-to-go estimates under the agent's actual (planned) behaviour,
    # which in turn improves the next round of planning.
    mode2_memory = ShortTermMemory()
    dataset = []
    for round_idx in range(1, 3):
        print(f"phase 3.{round_idx}: Mode-2 practice + critic refit")
        dataset += run_mode2_practice(
            agent, args.distill_episodes, mode2_memory,
            seed=args.seed + round_idx, tag=f"round {round_idx}",
        )
        refit_critic(agent, mode2_memory)

    print("phase 4: Mode-2 -> Mode-1 policy distillation")
    fit_policy(agent, dataset)

    save(agent)


if __name__ == "__main__":
    main()
