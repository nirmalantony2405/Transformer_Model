import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feedforward import PositionWiseFeedForward
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(embed_dim, feedforward_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src
