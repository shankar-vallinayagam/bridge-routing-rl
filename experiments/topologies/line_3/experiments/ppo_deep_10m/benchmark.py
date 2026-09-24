"""Run the shared line_3 benchmark harness for the deep 10M experiment."""
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    experiment_root = Path(__file__).resolve().parent
    shared_benchmark = experiment_root.parent / "ppo_baseline" / "benchmark.py"
    command = [
        sys.executable,
        str(shared_benchmark),
        "--experiment-dir",
        str(experiment_root),
    ] + sys.argv[1:]
    raise SystemExit(subprocess.call(command))
