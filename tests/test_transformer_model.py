import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer.modelling.transformer_model import TransformerModel

def test_transformer_model():
    vocab_size = 50
    d_model = 16
    n_heads = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 32
    dropout = 0.1
    max_len = 10
    
    model = TransformerModel(
        vocab_size, d_model, n_heads,
        num_encoder_layers, num_decoder_layers,
        dim_feedforward, dropout, max_len
    )
    
    src = torch.randint(0, vocab_size, (3, max_len))  
    tgt = torch.randint(0, vocab_size, (3, max_len))  
    src_mask = torch.ones((3, max_len)).bool()        
    tgt_mask = torch.ones((3, max_len)).bool()        
    
    output = model(src, tgt, src_mask, tgt_mask)
    assert output.shape == (3, max_len, vocab_size)

import torch

vocab_size = 5000  
d_model = 512  
n_heads = 8  
num_encoder_layers = 6  
num_decoder_layers = 6  
dim_feedforward = 2048  
dropout = 0.1  
max_len = 50  

model = TransformerModel(
    vocab_size, d_model, n_heads, num_encoder_layers,
    num_decoder_layers, dim_feedforward, dropout, max_len
)

batch_size = 4
src_len = 10  
tgt_len = 10 

src = torch.randint(0, vocab_size, (batch_size, src_len))
tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

src_mask = None  
tgt_mask = model.generate_square_subsequent_mask(tgt_len)
memory_mask = None

logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)
print("Logits shape:", logits.shape)

