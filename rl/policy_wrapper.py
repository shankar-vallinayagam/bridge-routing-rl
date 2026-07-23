import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

# Import your existing components
from models.feature_encoders import (
    QubitEmbedder,
    InteractionTableEncoder,
    GateSeqEmbedder,
    LayoutEmbedder,
    LayoutStateEmbedder,
    RoutingStateEmbedder
)
from models.action_heads import ActionHead
from models.basic_agent import (LayoutAgent, RoutingAgent)


class CircuitPolicy(ActorCriticPolicy):
    """
    Custom policy that integrates LayoutAgent and RoutingAgent with SB3's MaskablePPO.
    
    Uses two separate value networks for Layout and Routing phases.
    """
    
    def __init__(self, *args, Q=None, E=None, edge_list=None, qubit_embedding_dim=32, state_embedding_dim=256, action_hidden_dim=256, window_len=None, **kwargs):
        """
        Args:
            Q: Number of qubits
            E: Number of edges in the hardware architecture
            window_len: Length of the context window for gate sequences
        """
        # Store parameters before calling parent
        self.Q = Q
        self.E = E
        self.edge_list = edge_list
        self.window_len = window_len
        self.state_embedding_dim = state_embedding_dim
        
        # Call parent class
        super().__init__(*args, **kwargs)
        
        # Create Shared Embedders 
        self.logical_embedder = QubitEmbedder(Q + 1, qubit_embedding_dim)
        self.routing_embedder = QubitEmbedder(Q + 1, qubit_embedding_dim)

        # Layout-specific encoders
        self.gate_encoder_layout = GateSeqEmbedder(window_len, self.logical_embedder)
        self.layout_embedder = LayoutEmbedder(Q, self.logical_embedder, self.hardware_embedder)
        self.interaction_encoder_layout = InteractionTableEncoder(Q)
        
        # Routing-specific encoders
        self.gate_encoder_routing = GateSeqEmbedder(window_len, self.hardware_embedder)
        self.interaction_encoder_routing = InteractionTableEncoder(Q)
        
        # State Embedders 
        self.layout_state_embedder = LayoutStateEmbedder(
            self.interaction_encoder_layout,
            self.gate_encoder_layout,
            self.layout_embedder,
            state_embedding_dim
        )
        
        self.routing_state_embedder = RoutingStateEmbedder(
            self.interaction_encoder_routing,
            self.gate_encoder_routing,
            state_embedding_dim
        )
        
        # Create Agents 
        self.layout_agent = LayoutAgent(
            Q=Q,
            E=E,
            window_len=window_len,
            hardware_qubit_embedder=self.hardware_embedder,
            qubit_embedding_dim=qubit_embedding_dim,
            state_embedding_dim=state_embedding_dim,
            action_hidden_dim=action_hidden_dim,
            bridge_allowed=True
        )
        
        self.routing_agent = RoutingAgent(
            Q=Q,
            E=E,
            window_len=window_len,
            hardware_qubit_embedder=self.hardware_embedder,
            state_embedding_dim=state_embedding_dim,
            action_hidden_dim=action_hidden_dim,
            bridge_allowed=True
        )
        
        # Layout value network
        self.layout_value_net = nn.Sequential(
            nn.Linear(state_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Routing value network
        self.routing_value_net = nn.Sequential(
            nn.Linear(state_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Store total actions for later use
        self.total_actions = Q + E + 1

    def _get_logits_and_values(self, obs):
        """
        Shared method to compute logits and values from observations.
        
        Returns:
            logits: Action logits (batch_size, total_actions)
            values: State value estimates (batch_size,)
            is_layout: Boolean mask for layout phase
        """
        batch_size = obs["interaction_matrix"].shape[0]
        device = obs["interaction_matrix"].device
        
        total_actions = self.Q + self.E + 1
        logits = torch.full(
            (batch_size, total_actions),
            -1e9,
            device=device
        )
        values = torch.zeros(batch_size, device=device)
        
        is_layout = ~obs["layout_complete"].squeeze(-1).bool()
        
        # Layout phase
        if is_layout.any():
            idx = is_layout.nonzero().squeeze(1)
            logits[idx], values[idx] = self.layout_agent(
                obs["interaction_matrix"][idx],
                obs["context_window"][idx],
                obs["layout_table"][idx]
            )
        
        # Routing phase
        if (~is_layout).any():
            idx = (~is_layout).nonzero().squeeze(1)
            logits[idx], values[idx] = self.routing_agent(
                obs["interaction_matrix"][idx],
                obs["context_window"][idx]
            )
        
        return logits, values

    def forward(self, obs, deterministic=True):
        logits, values = self._get_logits_and_values(obs)
    
        dist = self._get_action_dist_from_latent(logits)
        actions = dist.get_actions(deterministic=deterministic)
        
        return actions, values, logits
        
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Observation dict
            actions: Actions taken
            
        Returns:
            values: State value estimates
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
        """

        logits, values = self._get_logits_and_values(obs)
    
        dist = self._get_action_dist_from_latent(logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_probs, entropy
        
    
    def get_action_mask(self, obs):
        """
        Get action mask for the current state.
        
        Args:
            obs: Observation dict from environment
            
        Returns:
            numpy array of shape (batch_size, action_space.n) with 1 for valid actions
        """
        batch_size = obs["interaction_matrix"].shape[0]
        layout_complete = obs["layout_complete"].squeeze(-1).cpu().numpy()
        top_gate = obs["context_window"][:, 0]
        
        masks = np.zeros((batch_size, self.total_actions), dtype=np.int8)
        
        for i in range(batch_size):
            if not layout_complete[i]:  # Layout phase
                masks[i, :self.Q] = 1
            else:  # Routing phase
                a, b = top_gate[i]
                if [a, b] in self.edge_list:
                    masks[i, self.edge_list.index([a, b])] = 1
                elif [b, a] in self.edge_list:
                    masks[i, self.edge_list.index([a, b])] = 1
                masks[i, self.Q+self.E] = 1  # Bridge action
        
        return masks