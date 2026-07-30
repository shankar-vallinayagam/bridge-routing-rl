"""
Minimal test that pipeline works
"""
from rl.ppo import TrainingConfig, AgentConfig, EnvConfig, train_model
from env.chip_architecture import ChipHardware

# 0 -- 1 -- 2 architecture
hardware_3q = ChipHardware(3, [[1], [0,2], [1]])

env_config = EnvConfig(
    hardware=hardware_3q,
    window_length=3,
    min_gate_count=1,
    max_gate_count=3,
)

agent_config = AgentConfig(
    state_embedding_dim=16,
    qubit_embedding_dim=4,
)

training_config = TrainingConfig(
    total_timesteps=512, 
    num_envs=2,
    num_steps=32,
    num_minibatches=2,
    update_epochs=1,
    track=False,          
    cuda=False,             
)

if __name__ == "__main__":
    train_model(training_config, agent_config, env_config)
    print("Test finished without crashing.")
