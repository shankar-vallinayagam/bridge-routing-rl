import torch
from torch import nn

class OutputHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, encoded_state):
        return self.head(encoded_state)


