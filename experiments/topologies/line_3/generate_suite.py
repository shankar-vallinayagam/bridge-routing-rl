"""Generate reproducible random CNOT benchmark cases for the line_3 topology."""
import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarking.cases import BenchmarkCase
from benchmarking.suite_io import write_cases
from benchmarking.topology import TopologySpec


def make_case(case_id, qubit_count, gate_count, seed):
    rng = random.Random(seed)
    gates = []
    for _ in range(gate_count):
        control = rng.randrange(qubit_count)
        target = rng.randrange(qubit_count - 1)
        if target >= control:
            target += 1
        gates.append(("cx", (control, target)))
    return BenchmarkCase(
        case_id=case_id,
        qubit_count=qubit_count,
        gates=tuple(gates),
        seed=seed,
        metadata={"depth_bucket": gate_count},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--per-depth", type=int, default=25)
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "suites" / "random_v1" / "cases.jsonl",
    )
    args = parser.parse_args()
    if args.per_depth < 1 or any(depth < 1 for depth in args.depths):
        raise ValueError("per-depth and every depth must be positive")

    topology = TopologySpec.from_json(Path(__file__).resolve().parent / "topology.json")
    cases = []
    for depth in args.depths:
        for index in range(args.per_depth):
            case_seed = args.seed + depth * 10000 + index
            cases.append(
                make_case(
                    f"random_depth_{depth}_{index:04d}",
                    topology.qubit_count,
                    depth,
                    case_seed,
                )
            )
    manifest = {
        "suite_id": "random_v1",
        "topology_id": topology.topology_id,
        "generator": "generate_suite.py",
        "seed": args.seed,
        "depths": args.depths,
        "cases_per_depth": args.per_depth,
        "case_count": len(cases),
    }
    write_cases(args.output, cases, manifest)
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
