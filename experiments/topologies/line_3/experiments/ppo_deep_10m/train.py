"""Run the shared line_3 PPO harness with the deep 10M configuration."""
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    experiment_root = Path(__file__).resolve().parent
    shared_train = experiment_root.parent / "ppo_baseline" / "train.py"
    command = [
        sys.executable,
        str(shared_train),
        "--experiment-dir",
        str(experiment_root),
    ] + sys.argv[1:]
    raise SystemExit(subprocess.call(command))
