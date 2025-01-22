import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from transformer_project.modelling.model import Attention
from transformer_project.modelling.model import PositionalEncoding
def test_attention_padding_mask():
    attention = Attention(d_model=512, num_heads=8)
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    mask = attention.create_padding_mask(seq)

    assert mask.size() == (2, 1, 1, 5)
    assert torch.equal(mask[0, 0, 0], torch.tensor([1, 1, 1, 0, 0]))
    assert torch.equal(mask[1, 0, 0], torch.tensor([1, 1, 0, 0, 0]))

def test_attention_future_mask():
    attention = Attention(d_model=512, num_heads=8)
    seq_length = 5
    mask = attention.create_future_mask(seq_length)

    assert mask.size() == (5, 5)
    assert torch.equal(mask, torch.tensor([
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
        [False, False, False, False, True],
        [False, False, False, False, False],
    ]))

def test_positional_encoding():
    pe = PositionalEncoding(d_model=16, max_len=100)
    input_tensor = torch.zeros((1, 100, 16))
    output_tensor = pe(input_tensor)
    assert output_tensor.shape == input_tensor.shape
    print("Positional encoding test passed!")