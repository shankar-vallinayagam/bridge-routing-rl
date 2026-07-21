import torch
from torch import nn

from feature_encoders import (
    QubitEmbedder,
    InteractionTableEncoder,
    GateSeqEmbedder,
    LayoutEmbedder,
    StateEmbedder,
)
from action_heads import ActionHead

class BasicAgent(nn.Module):
    def __init__(self, Q, E, window_len, qubit_embedding_dim, state_hidden_dim, state_embedding_dim, action_hidden_dim, bridge_allowed):
        super().__init__()
        self.Q = Q
        self.window_len = window_len
        self.bridge_allowed = bridge_allowed

        self.logical_qubit_embedder = QubitEmbedder(self.Q + 1, qubit_embedding_dim)
        self.hardware_qubit_embedder = QubitEmbedder(self.Q + 1, qubit_embedding_dim)

        self.interaction_table_embedder = InteractionTableEncoder(self.Q)
        self.gate_sequence_embedder = GateSeqEmbedder(self.window_len, self.logical_qubit_embedder)
        self.layout_embedder = LayoutEmbedder(self.Q, self.logical_qubit_embedder, self.hardware_qubit_embedder)

        self.state_embedder = StateEmbedder(
            self.interaction_table_embedder,
            self.gate_sequence_embedder,
            self.layout_embedder,
            state_hidden_dim,
            state_embedding_dim
            )

        self.layout_action_head = ActionHead(state_embedding_dim, action_hidden_dim, Q)
        self.routing_action_head = ActionHead(state_embedding_dim, action_hidden_dim, E+bridge_allowed)

    def forward(self, interaction_mat, gate_seq, layout_table, layout_complete):
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

            layout_mask = ~layout_complete
            routing_mask = layout_complete

            # Layout phase
            if layout_mask.any():
                layout_logits = self.layout_action_head(
                    state[layout_mask]
                )
                # fills first Q actions
                logits[layout_mask, :self.Q] = layout_logits

            # Routing phase
            if routing_mask.any():
                routing_logits = self.routing_action_head(
                    state[routing_mask]
                )
                # fills actions after Q
                logits[routing_mask, self.Q:] = routing_logits

            return logits


            







