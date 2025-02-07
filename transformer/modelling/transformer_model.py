import torch
import torch.nn as nn
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer.modelling.model import PositionalEncoding
from transformer.layers.transformer_encoder import TransformerEncoderLayer
from transformer.layers.transformer_decoder import TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model, 
        n_heads, 
        num_encoder_layers, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout, 
        max_len
    ):
        super(TransformerModel, self).__init__()

        # Embedding and Positional Encoding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Encoder and Decoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output Linear layer (shared with embedding layer)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.output_layer.weight = self.embedding.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Embedding + Positional Encoding for source
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.dropout(self.positional_encoding(src))

        # Embedding + Positional Encoding for target
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.dropout(self.positional_encoding(tgt))

        # Pass source through the encoder
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        # Pass target through the decoder
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        # Final output projection
        logits = self.output_layer(output)
        return logits
    
    def greedy_decode(self, src, src_mask, max_len, start_token, end_token):
        """
        Implementation of greedy decoding as described in the original code.
        """
        self.eval()  

        memory = self.encoder_layers(self.positional_encoding(self.embedding(src)), src_mask)

        # Start decoding with the start token
        tgt_tokens = torch.tensor([[start_token]], device=src.device)

        for _ in range(max_len):
            tgt_mask = self.generate_square_subsequent_mask(tgt_tokens.size(1)).to(src.device)
            tgt_emb = self.positional_encoding(self.embedding(tgt_tokens))
            output = self.decoder_layers(tgt_emb, memory, tgt_mask, src_mask)
            logits = self.output_layer(output[:, -1, :])
            next_token = logits.argmax(dim=-1).item()
            tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]], device=src.device)], dim=1)

            if next_token == end_token:
                break

        return tgt_tokens.squeeze(0).tolist()