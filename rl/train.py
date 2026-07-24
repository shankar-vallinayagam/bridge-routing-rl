import random
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.chip_architecture import ChipHardware
from env.quantum_circuit import GateSequence
from env.environment import CircuitEnvironment
from rl.policy_wrapper import CircuitPolicy


# Configs

@dataclass
class EnvConfig:
    hardware: ChipHardware
    window_length: int = 5
    min_gate_count: int = 1
    max_gate_count: int = 10
    seed: int = 42

@dataclass
class TrainConfig:
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    policy_kwargs: Dict[str, Any]= field(default_factory=dict)
    log_dir: str = "./logs/tensorboard/"
    model_dir: str = "./logs/models/"
    seed: int = 42

def make_env(env_config: EnvConfig) -> CircuitEnvironment:
    """Create a CircuitEnvironment wrapped with ActionMasker."""
    architecture = env_config.hardware
    env = CircuitEnvironment(
        architecture=architecture,
        window_length=env_config.window_length,
        min_gate_count=env_config.min_gate_count,
        max_gate_count=env_config.max_gate_count,
    )
    if env_config.seed is not None:
        env.seed(env_config.seed)
    env = ActionMasker(env, lambda e: e.get_action_mask())
    return env

def make_vec_env(env_config: EnvConfig, n_envs: int = 1):
    """Create a vectorised environment (single or parallel)."""
    def _init():
        return make_env(env_config)
    if n_envs == 1:
        return DummyVecEnv([_init])
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        return SubprocVecEnv([_init for _ in range(n_envs)])

def create_model(env_config: EnvConfig, train_config: TrainConfig, vec_env):
    Q = env_config.hardware.qubits
    E = len(env_config.hardware.edges)
    edges = env_config.hardware.edges

    # default kwargs
    policy_kwargs = {
        "Q": Q,
        "E": E,
        "window_len": env_config.window_length,
        "edge_list": edges,
        "state_embedding_dim": 128,
        "action_hidden_dim": 64,
        "qubit_embedding_dim": 32,
    }

    policy_kwargs.update(train_config.policy_kwargs)

    model = MaskablePPO(
            policy=CircuitPolicy,
            env=vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=train_config.learning_rate,
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            n_epochs=train_config.n_epochs,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            clip_range=train_config.clip_range,
            ent_coef=train_config.ent_coef,
            verbose=1,
            tensorboard_log=train_config.log_dir,
            seed=train_config.seed,
        )
    
    return model 

def train(
    env_config: EnvConfig,
    train_config: TrainConfig,
    name: str,
    n_envs=1
):
    """
    Run the RL training with the given configurations.
    Returns the trained model.
    """

    vec_env = make_vec_env(env_config, n_envs=n_envs)

    model = create_model(env_config, train_config, vec_env)

    print("Starting training...")
    model.learn(
        total_timesteps=train_config.total_timesteps,
        log_interval=50
    )

    Path(train_config.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = f"{train_config.model_dir}/{name}"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    return model