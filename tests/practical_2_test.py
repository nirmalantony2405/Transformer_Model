import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from transformer.modelling.model import Attention

def test_attention_mechanism():
    attention = Attention(d_model=512, num_heads=8)

    query = torch.rand(2, 5, 512)
    key = torch.rand(2, 5, 512)
    value = torch.rand(2, 5, 512)

    output = attention(query, key, value)
    assert output.size() == (2, 5, 512)

    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    pad_mask = attention.create_padding_mask(seq)
    output_with_pad_mask = attention(query, key, value, mask=pad_mask)
    assert output_with_pad_mask.size() == (2, 5, 512)

    future_mask = attention.create_future_mask(seq_length=5)
    output_with_future_mask = attention(query, key, value, mask=future_mask.unsqueeze(0))
    assert output_with_future_mask.size() == (2, 5, 512)
