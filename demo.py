"""Demo: the world-model agent writes code by planning in latent space.

Give it a spec (a target function); the configurator turns the spec into
a goal, the actor imagines edit sequences with the JEPA world model
(Mode-2 reasoning — no program is executed during planning), and the
agent writes the program. The written code is then verified against
held-out test inputs the agent never perceives.

Usage:
    python demo.py                       # run the evaluation suite
    python demo.py --spec "a * b + a"    # write code for a custom spec
    python demo.py --mode 1              # use the distilled reactive policy
"""

import argparse

from alemayhu.agent import Agent
from alemayhu.train import load

# Held-out tests: inputs disjoint from the probe inputs the agent observes.
HOLDOUT_TESTS = [(5.0, 7.0), (-2.0, 9.0), (1.5, 2.5), (-4.0, -6.0), (0.0, 0.0), (10.0, 3.0)]

SUITE = [
    ("a + b", lambda a, b: a + b),
    ("a * b", lambda a, b: a * b),
    ("a - b", lambda a, b: a - b),
    ("(a + b) * 2", lambda a, b: (a + b) * 2),
    ("a * a", lambda a, b: a * a),
    ("2 * a + b", lambda a, b: 2 * a + b),
    ("max(a, b)", lambda a, b: max(a, b)),
    ("min(a, b) + 1", lambda a, b: min(a, b) + 1),
    ("abs(a - b)", lambda a, b: abs(a - b)),
    ("a * b + a", lambda a, b: a * b + a),
    ("-(a + b)", lambda a, b: -(a + b)),
    ("(a - b) * (a - b)", lambda a, b: (a - b) * (a - b)),
]


def verify(fn, spec) -> bool:
    for a, b in HOLDOUT_TESTS:
        out = fn(a, b)
        if out is None or abs(out - spec(a, b)) > 1e-6:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=str, default=None,
                        help="expression in a, b — e.g. 'a * b + a'")
    parser.add_argument("--mode", type=int, default=2, choices=[1, 2],
                        help="2 = plan with world model, 1 = reactive policy")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    agent = load(Agent())

    if args.spec:
        spec = eval(f"lambda a, b: {args.spec}", {"max": max, "min": min, "abs": abs})
        tasks = [(args.spec, spec)]
    else:
        tasks = SUITE

    mode_name = "Mode-2 (planning)" if args.mode == 2 else "Mode-1 (reactive)"
    print(f"writing code with {mode_name}\n")

    passed = 0
    for name, spec in tasks:
        result = agent.write_code(spec, mode=args.mode, verbose=args.verbose)
        ok = verify(result.fn, spec)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] spec: f(a, b) = {name}   "
              f"(final intrinsic energy {result.final_energy:.4f})")
        print("\n".join("    " + line for line in result.program.splitlines()))
        print()

    print(f"{passed}/{len(tasks)} specs passed held-out tests")


if __name__ == "__main__":
    main()
