"""Benchmark a line_3 checkpoint on a saved suite."""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from benchmarking.adapters.random_router import RandomRouterAdapter
from benchmarking.adapters.trained_agent import TrainedAgentAdapter
from benchmarking.runner import run_benchmark
from benchmarking.suite_io import load_cases
from benchmarking.topology import TopologySpec


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOPOLOGY_ROOT = DEFAULT_EXPERIMENT_ROOT.parents[1]


def latest_checkpoint(experiment_root):
    candidates = [
        path for path in (experiment_root / "checkpoints").glob("*.pt")
        if "_iter" not in path.stem
    ]
    if not candidates:
        raise FileNotFoundError("no final checkpoint found; train ppo_baseline first")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--suite",
        type=Path,
        default=TOPOLOGY_ROOT / "suites" / "random_v1" / "cases.jsonl",
    )
    parser.add_argument("--methods", default="agent,random")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-decisions", type=int, default=10000)
    args = parser.parse_args()

    experiment_root = args.experiment_dir.resolve()
    topology = TopologySpec.from_json(TOPOLOGY_ROOT / "topology.json")
    cases = load_cases(args.suite)
    methods = {name.strip() for name in args.methods.split(",") if name.strip()}
    checkpoint = args.checkpoint or (latest_checkpoint(experiment_root) if "agent" in methods else None)
    adapters = []
    if "agent" in methods:
        adapters.append(
            TrainedAgentAdapter(
                checkpoint,
                device=args.device,
                max_decisions=args.max_decisions,
            )
        )
    if "random" in methods:
        adapters.append(RandomRouterAdapter(topology.chip_hardware(), max_decisions=args.max_decisions))
    if "sabre" in methods:
        from benchmarking.adapters.qiskit_sabre import QiskitSabreAdapter

        adapters.append(QiskitSabreAdapter(topology))
    if not adapters:
        raise ValueError("choose at least one method: agent, random, sabre")

    run_id = checkpoint.stem if "agent" in methods else "no_agent"
    output_dir = experiment_root / "benchmark_results" / run_id
    results = run_benchmark(cases, adapters, output_dir, seeds=(args.seed,))

    summary = defaultdict(list)
    for result in results:
        if result["success"] and result["added_cnot_count"] is not None:
            summary[result["method"]].append(result["added_cnot_count"])
    print(f"Results written to {output_dir}")
    for method, values in sorted(summary.items()):
        print(
            f"{method}: cases={len(values)}, mean_added_cnot={sum(values) / len(values):.3f}, "
            f"min={min(values)}, max={max(values)}"
        )


if __name__ == "__main__":
    main()
