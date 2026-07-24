from env.chip_architecture import ChipHardware
from rl.train import train, EnvConfig, TrainConfig

# 3-qubit line
hardware = ChipHardware(qubit_count=3, adj_list=[[0,1], [1,2]])

env_config = EnvConfig(
    hardware=hardware,
    window_length=3,
    min_gate_count=1,
    max_gate_count=5,
)

train_config = TrainConfig(
    total_timesteps=2_000,
    n_steps=512,
    batch_size=16,
    n_epochs=3,
    policy_kwargs={
        "state_embedding_dim": 16,
        "action_hidden_dim": 8,
        "qubit_embedding_dim": 4,
    },
)

model = train(env_config, train_config, name="3q_test_model")
print("Done!")