import torch
from torch import nn

from feature_encoders import (
    QubitEmbedder,
    InteractionTableEncoder,
    GateSeqEmbedder,
    LayoutEmbedder,
    LayoutStateEmbedder,
    RoutingStateEmbedder
)
from action_heads import ActionHead

class LayoutAgent(nn.Module):
    def __init__(self, Q, E, window_len, hardware_qubit_embedder: QubitEmbedder, qubit_embedding_dim, state_embedding_dim, action_hidden_dim, bridge_allowed):
        super().__init__()
        self.Q = Q
        self.E = E
        self.window_len = window_len
        self.bridge_allowed = bridge_allowed

        self.logical_qubit_embedder = QubitEmbedder(self.Q + 1, qubit_embedding_dim)
        self.hardware_qubit_embedder = hardware_qubit_embedder

        self.interaction_table_embedder = InteractionTableEncoder(self.Q)
        self.gate_sequence_embedder = GateSeqEmbedder(self.window_len, self.logical_qubit_embedder)
        self.layout_embedder = LayoutEmbedder(self.Q, self.logical_qubit_embedder, self.hardware_qubit_embedder)

        self.state_embedder = LayoutStateEmbedder(
            self.interaction_table_embedder,
            self.gate_sequence_embedder,
            self.layout_embedder,
            state_embedding_dim
            )

        self.layout_action_head = ActionHead(state_embedding_dim, action_hidden_dim, Q)

        self.layout_value_head = nn.Sequential(
            nn.Linear(state_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, interaction_mat, gate_seq, layout_table):
            state = self.state_embedder(
                interaction_mat,
                gate_seq,
                layout_table
            )
            
            batch_size = state.shape[0]
            total_actions = self.Q + self.E + 1

            # final PPO logits
            logits = torch.full(
                (batch_size, total_actions),
                -1e9,
                device=state.device
            )

            layout_logits = self.layout_action_head(state)
            # fills first Q actions
            logits[:, :self.Q] = layout_logits

            value = self.value_head(state).squeeze(-1)

            return logits, value

class RoutingAgent(nn.module):

    def __init__(self, Q, E, window_len, hardware_qubit_embedder: QubitEmbedder, state_embedding_dim, action_hidden_dim, bridge_allowed):
            super().__init__()
            self.Q = Q
            self.E = E
            self.window_len = window_len
            self.bridge_allowed = bridge_allowed

            self.hardware_qubit_embedder = hardware_qubit_embedder

            self.interaction_table_embedder = InteractionTableEncoder(self.Q)
            self.gate_sequence_embedder = GateSeqEmbedder(self.window_len, self.hardware_qubit_embedder)

            self.state_embedder = RoutingStateEmbedder(
                self.interaction_table_embedder,
                self.gate_sequence_embedder,
                state_embedding_dim
                )

            self.routing_action_head = ActionHead(state_embedding_dim, action_hidden_dim, Q)
            self.routing_value_head = nn.Sequential(
                nn.Linear(state_embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )


    def forward(self, interaction_mat, gate_seq):
            state = self.state_embedder(
                interaction_mat,
                gate_seq
            )
            
            batch_size = state.shape[0]
            total_actions = self.Q + self.E + 1

            # final PPO logits
            logits = torch.full(
                (batch_size, total_actions),
                -1e9,
                device=state.device
            )

            routing_logits = self.routing_action_head(state)
            # fills first Q actions
            logits[:, :self.Q] = routing_logits

            value = self.value_head(state).squeeze(-1)

            return logits, value
