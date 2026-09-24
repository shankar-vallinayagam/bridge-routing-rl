"""Train PPO seriously on a three-qubit linear device.

TensorBoard event files are written by ``train_model`` below ``runs/``.  From
the repository root, monitor this experiment with::

    tensorboard --logdir runs

The defaults are intended for a real training run.  Use the command-line
options to shorten or scale the run for a particular machine.
"""
import argparse
import sys
from pathlib import Path

if sys.version_info < (3, 9):
    raise RuntimeError(
        "Python 3.9 or newer is required. Run this project with "
        "'.venv/bin/python' instead of the system Python 3.8 interpreter."
    )

# Support both `python experiments/train_3q.py` and
# `python -m experiments.train_3q` from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.chip_architecture import ChipHardware
from rl.ppo import AgentConfig, EnvConfig, TrainingConfig, train_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-name", default="train_3q_serious")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--window-length", type=int, default=12)
    parser.add_argument("--min-gates", type=int, default=5)
    parser.add_argument("--max-gates", type=int, default=20)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--qubit-dim", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU even when CUDA is available",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.num_envs * args.num_steps
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) < 1:
        raise ValueError("environment, rollout, minibatch, and update counts must be positive")
    if args.total_timesteps < batch_size:
        raise ValueError("total-timesteps must be at least num-envs * num-steps")
    if args.num_minibatches > batch_size:
        raise ValueError("num-minibatches cannot exceed num-envs * num-steps")
    if batch_size % args.num_minibatches != 0:
        raise ValueError("num-envs * num-steps must be divisible by num-minibatches")
    if args.min_gates < 1 or args.max_gates < args.min_gates:
        raise ValueError("max-gates must be at least min-gates, and min-gates must be positive")

    # 0 -- 1 -- 2: a minimal but non-trivial nearest-neighbour device.
    hardware_3q = ChipHardware(3, [[1], [0, 2], [1]])

    env_config = EnvConfig(
        hardware=hardware_3q,
        window_length=args.window_length,
        min_gate_count=args.min_gates,
        max_gate_count=args.max_gates,
    )
    agent_config = AgentConfig(
        state_embedding_dim=args.state_dim,
        qubit_embedding_dim=args.qubit_dim,
    )
    training_config = TrainingConfig(
        exp_name=args.exp_name,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        checkpoint_interval=args.checkpoint_interval,
        track=False,
        cuda=not args.cpu,
    )

    run_name = train_model(training_config, agent_config, env_config)
    print(f"Training complete: {run_name}")
    print("TensorBoard: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
