"""
This is a modified form of the ppo.py script from CleanRL
"""
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
import pickle

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from models.output_heads import OutputHead
from models.feature_encoders import (QubitEmbedder,
                                     LayoutStateEmbedder,
                                     RoutingStateEmbedder,
                                     GateSeqEmbedder,
                                     LayoutEmbedder,
                                     InteractionTableEncoder)
from env.chip_architecture import (ChipHardware)
from env.environment import CircuitEnvironment

@dataclass
class AgentConfig:
    state_embedding_dim: int = 128
    qubit_embedding_dim: int = 16

@dataclass 
class EnvConfig:
    hardware: ChipHardware = None
    window_length: int = 30
    min_gate_count: int = 1
    max_gate_count: int = 10


@dataclass
class TrainingConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    checkpoint_interval: int = 50
    """The number of iterations to save a checkpoint (in case of crash on long runs)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, agent_config: AgentConfig, env_config: EnvConfig):
        super().__init__()

        self.hardware = env_config.hardware
        self.window_length = env_config.window_length
        self.Q = self.hardware.qubit_count
        self.E = self.hardware.edge_count

        self.qubit_embedding_dimension = agent_config.qubit_embedding_dim
        self.state_embedding_dimension = agent_config.state_embedding_dim

        self.logical_embedder = QubitEmbedder(self.Q + 1, self.qubit_embedding_dimension)
        self.hardware_embedder = QubitEmbedder(self.Q + 1, self.qubit_embedding_dimension)

        self.gate_encoder_layout = GateSeqEmbedder(self.window_length, self.logical_embedder)
        self.interaction_table_layout = InteractionTableEncoder(self.Q)
        self.layout_embedder = LayoutEmbedder(self.Q, self.logical_embedder, self.hardware_embedder)

        self.gate_encoder_routing = GateSeqEmbedder(self.window_length, self.hardware_embedder)
        self.interaction_table_routing = InteractionTableEncoder(self.Q)

        self.layout_state_embedder = LayoutStateEmbedder(self.interaction_table_layout,
                                                          self.gate_encoder_layout,
                                                          self.layout_embedder,
                                                          self.state_embedding_dimension)
        self.routing_state_embedder = RoutingStateEmbedder(self.interaction_table_routing,
                                                            self.gate_encoder_routing,
                                                            self.state_embedding_dimension)

        self.layout_critic = OutputHead(self.state_embedding_dimension, 1)
        self.routing_critic = OutputHead(self.state_embedding_dimension, 1)

        self.layout_actor = OutputHead(self.state_embedding_dimension, self.Q + self.E + 1)
        self.routing_actor = OutputHead(self.state_embedding_dimension, self.Q + self.E + 1)

    def get_value(self, obs_dict):
        routing_mask = obs_dict["layout_complete"]  # tensor, shape (batch,), 0=layout 1=routing
        layout_mask = ~routing_mask

        values = torch.zeros(routing_mask.shape[0], 1, device=routing_mask.device)
        if layout_mask.any():
            layout_state = self.layout_state_embedder(obs_dict)
            values[layout_mask] = self.layout_critic(layout_state)
        if routing_mask.any():
            routing_state = self.routing_state_embedder(obs_dict)
            values[routing_mask] = self.routing_critic(routing_state)
        return values

    def get_action_and_value(self, obs_dict, action=None, deterministic=False):
        routing_batch_mask = obs_dict["layout_complete"]
        layout_batch_mask = ~routing_batch_mask

        batch = routing_batch_mask.shape[0]
        logprob = torch.zeros(batch, device=routing_batch_mask.device)
        entropy = torch.zeros(batch, device=routing_batch_mask.device)
        value = torch.zeros(batch, 1, device=routing_batch_mask.device)
        out_action = torch.zeros(batch, dtype=torch.long, device=routing_batch_mask.device) if action is None else action

        for mask, state_embedder, actor, critic, is_layout in [
            (layout_batch_mask, self.layout_state_embedder, self.layout_actor, self.layout_critic, True),
            (routing_batch_mask, self.routing_state_embedder, self.routing_actor, self.routing_critic, False),
        ]:
            if not mask.any():
                continue

            state = state_embedder(obs_dict)
            logits = actor(state)  # (batch, Q + E + 1)

            action_mask = torch.zeros(mask.sum(), self.Q + self.E + 1, dtype=torch.bool, device=mask.device)
            if is_layout:
                layout_table = obs_dict["layout_table"][mask]                      # (batch, K), padded with self.Q
                valid = torch.ones(layout_table.shape[0], self.Q + 1, dtype=torch.bool, device=mask.device)
                valid.scatter_(1, layout_table, False)
                action_mask[:, :self.Q] = valid[:, :self.Q]
            else:
                leading = obs_dict["context_window"][mask]                        # (batch, 2) physical qubit pair
                q0, q1 = leading[:, 0], leading[:, 1]
                valid_edges = self.hardware.incidence_table[q0] | self.hardware.incidence_table[q1]  # (n_sub, E)
                action_mask[:, self.Q:self.Q + self.E] = valid_edges

            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
            dist = Categorical(logits=logits)

            if action is not None:
                a = action[mask]
            elif deterministic:
                a = dist.probs.argmax(dim=-1)
            else:
                a = dist.sample()


            if action is None:
                out_action[mask] = a
    
            logprob[mask] = dist.log_prob(a)
            entropy[mask] = dist.entropy()
            value[mask] = critic(state)

        return out_action, logprob, entropy, value

def make_env(env_config: EnvConfig):
    def thunk():
        env = CircuitEnvironment(
            architecture=env_config.hardware,
            window_length=env_config.window_length,
            min_gate_count=env_config.min_gate_count,
            max_gate_count=env_config.max_gate_count
        )
    return thunk
        
def train_model(training_config: TrainingConfig, agent_config: AgentConfig, env_config: EnvConfig):
    training_config.batch_size = int(training_config.num_envs * training_config.num_steps)
    training_config.minibatch_size = int(training_config.batch_size // training_config.num_minibatches)
    training_config.num_iterations = training_config.total_timesteps // training_config.batch_size
    run_name = f"{training_config.exp_name}__{training_config.seed}__{int(time.time())}"
    if training_config.track:
        import wandb

        wandb.init(
            project=training_config.wandb_project_name,
            entity=training_config.wandb_entity,
            sync_tensorboard=True,
            config=vars(training_config),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(training_config).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    torch.backends.cudnn.deterministic = training_config.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and training_config.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_config) for i in range(training_config.num_envs)],
    )

    agent = Agent(agent_config, env_config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=training_config.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = {
        k: torch.zeros((training_config.num_steps, training_config.num_envs) + v.shape).to(device)
        for k, v in envs.single_observation_space.spaces.items()
    }
    actions = torch.zeros((training_config.num_steps, training_config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((training_config.num_steps, training_config.num_envs)).to(device)
    rewards = torch.zeros((training_config.num_steps, training_config.num_envs)).to(device)
    dones = torch.zeros((training_config.num_steps, training_config.num_envs)).to(device)
    values = torch.zeros((training_config.num_steps, training_config.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=training_config.seed)
    next_obs = {k: torch.as_tensor(v).to(device) for k, v in next_obs.items()}
    next_done = torch.zeros(training_config.num_envs).to(device)

    for iteration in range(1, training_config.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if training_config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / training_config.num_iterations
            lrnow = frac * training_config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, training_config.num_steps):
            global_step += training_config.num_envs
            for k in obs:
                obs[k][step] = next_obs[k]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = {k: torch.as_tensor(v).to(device) for k, v in next_obs.items()}
            next_done = torch.as_tensor(next_done).to(device)


            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(training_config.num_steps)):
                if t == training_config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + training_config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + training_config.gamma * training_config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = {k: v.reshape((-1,) + v.shape[2:]) for k, v in obs.items()}
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(training_config.batch_size)
        clipfracs = []
        for epoch in range(training_config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, training_config.batch_size, training_config.minibatch_size):
                end = start + training_config.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > training_config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if training_config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - training_config.clip_coef, 1 + training_config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if training_config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -training_config.clip_coef,
                        training_config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - training_config.ent_coef * entropy_loss + v_loss * training_config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), training_config.max_grad_norm)
                optimizer.step()

            if training_config.target_kl is not None and approx_kl > training_config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Regular checkpoints
        os.makedirs("checkpoints", exist_ok=True)
        if iteration % training_config.checkpoint_interval == 0:
            torch.save(agent.state_dict(), f"checkpoints/{run_name}_iter{iteration}.pt")

    
    torch.save(agent.state_dict(), f"checkpoints/{run_name}.pt")
    with open(f"checkpoints/{run_name}_config.pkl", "wb") as f:
        pickle.dump({"agent_config": agent_config, "env_config": env_config}, f)



    envs.close()
    writer.close()

