import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.feed_forward(x)