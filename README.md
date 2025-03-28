# Implementing Machine Translation Transformer Model

## project Overview
This repository contains the implementation of a transformer model inspired by the architecture introduced in the seminal paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). The project showcases a step-by-step approach to constructing and training transformers for natural language processing (NLP) tasks. It also includes experimental analyses comparing GPU and CPU training efficiencies.

## Project Structure
```
Transformer_model/
├── README.md
├── model_evaluation
│  ├──greedy_decode.py
│  └── evaluation.py
├── tests
├── transformer
│  ├── data
│    ├── dataset.py
│    └── main.py
│  ├── layers
│    ├── feedforward.py
│    ├── multi_head_attention.py
│    ├── transformer_decoder.py
│    └── transformer_encoder.py
│  ├── modelling
│    ├── model.py
│    └── transformer_model.py
│  ├── schedulers
│    ├── LR_scheduler.py
│    └── adamw_optimizer.py
│  ├── tokenization
│    ├── bpe_tokenizer.py
│    └── hf_bpe_tokenizer.py
│  └── training 
│    ├── transformer_model_training.py
└── requirements.txt
```

## Overview

Transformers have revolutionized NLP by leveraging self-attention mechanisms to process sequential data without relying on recurrence. This repository implements a transformer model from scratch, including the following:

- Byte-Pair Encoding (BPE) tokenization
- Multi-head attention
- Positional encoding
- Feedforward layers
- Encoder and decoder layers
- Full transformer architecture
- Custom optimizer and learning rate scheduler
- transformer model training
- Cpu vs GPU training 
- model evaluation  : Machine translation between German    and English and BLEU score evaluation 

## Installation

### Prerequisites
- Python 3.11+
- poetry 
- PyTorch
- Hugging Face Transformers
- evaluate library

### Setup
```bash
git clone git@github.com/nirmalantony2405/Transformer_Model.git
cd Transformer_Model
pip install -r requirements.txt
```

## Hyperparameters
- Model Dimension: 64
- Number of Heads: 4
- Encoder Layers: 2
- Decoder Layers: 2
- Dropout: 0.1

## Performance Metrics
- Training Loss: 5.2400
- Validation Loss: 7.2606

## Known Issues
- BLEU score is 0.2873
- Not Exact Translation but which is close to the source

## References
1. Vaswani et al. "Attention is All You Need" (2017)
2. PyTorch Transformer Documentation
3. Hugging Face Transformers Library

