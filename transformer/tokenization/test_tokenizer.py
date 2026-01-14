import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer.tokenization.bpe_tokenizer import BPETokenizer
from transformer.tokenization.hf_bpe_tokenizer import HuggingfaceBPETokenizer

corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
]

sentence = "Machine learning is a subset of artificial intelligence."

custom_bpe = BPETokenizer(vocab_size=64)
custom_bpe.get_base_vocab(corpus)
custom_bpe.train(corpus)
custom_tokens = custom_bpe.tokenize(sentence)
print("Custom BPE Tokenized Sentence:", custom_tokens)

hf_bpe = HuggingfaceBPETokenizer(vocab_size=295)
hf_bpe.train(corpus)
hf_tokens = hf_bpe.tokenize(sentence)
print("Huggingface BPE Tokenized Sentence:", hf_tokens)
