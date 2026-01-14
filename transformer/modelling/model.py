import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / d_model)))

        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)

    @staticmethod
    def create_padding_mask(seq, pad_token=0):
        """
        Create a binary mask to ignore padding tokens.
        """
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def create_future_mask(seq_length):
        """
        Create a mask to prevent attention to future tokens.
        """
        return torch.triu(torch.ones((seq_length, seq_length), device='cuda' if torch.cuda.is_available() else 'cpu'), diagonal=1).bool()


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = Attention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_out = self.attention(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attention_out))

        ffn_out = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_out))

        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_heads, num_layers, ff_hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)

        x = self.positional_encoding(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        x = self.output_layer(x)
        return x
    
def clean_sentence(sentence):
    whitelist = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    sentence = re.sub(r'[^{}]'.format(re.escape(whitelist)), '', sentence)
    sentence = sentence.lower()
    return sentence

def clean_dataset(dataset, min_length=5, max_length=64):
    cleaned_data = []
    for pair in dataset:
        source, target = pair['translation']['de'], pair['translation']['en']
        source, target = clean_sentence(source), clean_sentence(target)
        
        if (min_length <= len(source.split()) <= max_length and 
            min_length <= len(target.split()) <= max_length and 
            len(source.split()) / len(target.split()) <= 2):
            cleaned_data.append({'de': source, 'en': target})
    return cleaned_data

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]    
