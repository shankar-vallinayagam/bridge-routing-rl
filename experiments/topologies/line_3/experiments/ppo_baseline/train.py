"""Train the baseline PPO agent with outputs local to this experiment."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from env.chip_architecture import ChipHardware
from rl.ppo import AgentConfig, EnvConfig, TrainingConfig, train_model
from benchmarking.topology import TopologySpec


DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOPOLOGY_ROOT = DEFAULT_EXPERIMENT_ROOT.parents[1]


def main():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    bootstrap_parser.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap_parser.parse_known_args()
    experiment_root = bootstrap_args.experiment_dir.resolve()
    config_path = (bootstrap_args.config or experiment_root / "config.json").resolve()

    with config_path.open() as handle:
        defaults = json.load(handle)
    parser = argparse.ArgumentParser(parents=[bootstrap_parser], description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=defaults["total_timesteps"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=defaults["num_envs"])
    parser.add_argument("--num-steps", type=int, default=defaults["num_steps"])
    parser.add_argument("--num-minibatches", type=int, default=defaults["num_minibatches"])
    parser.add_argument("--update-epochs", type=int, default=defaults["update_epochs"])
    parser.add_argument("--window-length", type=int, default=defaults["window_length"])
    parser.add_argument("--min-gates", type=int, default=defaults["min_gates"])
    parser.add_argument("--max-gates", type=int, default=defaults["max_gates"])
    parser.add_argument("--state-dim", type=int, default=defaults["state_dim"])
    parser.add_argument("--qubit-dim", type=int, default=defaults["qubit_dim"])
    parser.add_argument("--checkpoint-interval", type=int, default=defaults["checkpoint_interval"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(remaining)

    topology = TopologySpec.from_json(TOPOLOGY_ROOT / "topology.json")
    hardware = ChipHardware(topology.qubit_count, topology.adjacency)
    env_config = EnvConfig(
        hardware=hardware,
        window_length=args.window_length,
        min_gate_count=args.min_gates,
        max_gate_count=args.max_gates,
    )
    agent_config = AgentConfig(
        state_embedding_dim=args.state_dim,
        qubit_embedding_dim=args.qubit_dim,
    )
    training_config = TrainingConfig(
        exp_name=defaults.get("experiment_id", experiment_root.name),
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        checkpoint_interval=args.checkpoint_interval,
        track=False,
        cuda=not args.cpu,
        output_dir=str(experiment_root),
    )
    run_name = train_model(training_config, agent_config, env_config)
    print(f"Training complete: {experiment_root / 'checkpoints' / (run_name + '.pt')}")


if __name__ == "__main__":
    main()
