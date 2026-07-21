import torch 
from torch import nn

class InteractionTableEncoder(nn.Module):
    '''Flattens an interaction table, only upper triangular elts'''
    def __init__(self, Q):
        super().__init__()
        self.Q = Q

        # relevant upper triangular indices, save to buffer
        row_idx, col_idx = torch.triu_indices(Q, Q, offset=1)
        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)

        self.output_dim = Q*(Q-1)//2

    def forward(self, interaction_mat):
        # first index handles batches
        # use sqrt so large interaction counts are slightly truncated
        return torch.sqrt(interaction_mat[..., self.row_idx, self.col_idx])
    
class QubitEmbedder(nn.Module):
    '''Converts Qubits to an embedding.'''
    def __init__(self, Q, embedding_dim):
        super().__init__()
        self.Q = Q
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=self.Q,
            embedding_dim=self.embedding_dim
        )

    def forward(self, qubit_id):
        return self.embedding(qubit_id)
    
class GateSeqEmbedder(nn.Module):
    def __init__(self, window_length, qubit_embedder: QubitEmbedder):
        super().__init__()
        self.window_length = window_length
        self.embedding = qubit_embedder

        self.output_dim = qubit_embedder.embedding_dim * self.window_length * 2

    def forward(self, gate_seq):
        '''gate_seq has dims (batch_size, window_length, 2) when input'''

        # convert to tensor
        seq_tensor = torch.as_tensor(gate_seq, device=self.embedding.embedding.weight.device, dtype=torch.long)

        #embed
        raw_embeddings = self.embedding(seq_tensor)

        # concatenate pairs (so (1,2) goes to [embedding_1_1, ..., embedding_1_d, embedding_2_1, ..., embedding_2_d])
        # dimension is (batch_size, window_length*2*embedding_dim) when output
        return raw_embeddings.flatten(start_dim=1)
    
class LayoutEmbedder(nn.Module):
    '''Embeds the layout table; note hqe has Q+1 ids with the last referring to unnasigned qubits'''
    def __init__(self, Q, logical_qubit_embedder: QubitEmbedder, hardware_qubit_embedder: QubitEmbedder):
        super().__init__()
        self.Q = Q
        self.lqe = logical_qubit_embedder
        self.hqe = hardware_qubit_embedder

        self.output_dim = Q * (self.lqe.embedding_dim + self.hqe.embedding_dim)
    
    def forward(self, layout_table):
        layout_tensor = torch.as_tensor(
        layout_table,
        device=self.lqe.embedding.weight.device,
        dtype=torch.long
        )

        batch_size = layout_tensor.shape[0]

        logical_ids = torch.arange(self.Q, device=layout_tensor.device)

        # (Q, ld)
        logical_embeddings = self.lqe(logical_ids)

        # expand to (batch, Q, ld)
        logical_embeddings = logical_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # (batch, Q, hd)
        hardware_embeddings = self.hqe(layout_tensor)

        # (batch, Q, ld+hd)
        layout_embeddings = torch.cat([logical_embeddings, hardware_embeddings], dim=-1)

        # (batch, Q*(ld+hd))
        return layout_embeddings.flatten(start_dim=1)

    
class StateEmbedder(nn.Module):
    '''Takes in all relevant information regarding the state; Interaction table, context window, layout embedding,
    passes through 1 hidden layer, then finally embeds state'''
    def __init__(self, itf: InteractionTableEncoder, gse: GateSeqEmbedder,le: LayoutEmbedder, hidden_dim, embedding_dim):
        super().__init__()
        self.itf = itf
        self.gse = gse
        self.le = le
        self.shared_trunk = nn.Sequential(
            nn.Linear(self.itf.output_dim + self.gse.output_dim + self.le.output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self, interaction_mat, gate_seq, layout_table):
        flattened_mat = self.itf(interaction_mat)
        embedded_seq = self.gse(gate_seq)
        embedded_layout = self.le(layout_table)

        # concatenate states along content dimension (batch dim preserved)
        concat_state = torch.cat([flattened_mat, embedded_seq, embedded_layout], dim=1)
        return self.shared_trunk(concat_state)
        


# TODO: Later, add attention encoder to be swapped out for LinearEncoder to encode the output of GateSeqEmbedder,
# TODO: Add GNN encoder for interaction mat