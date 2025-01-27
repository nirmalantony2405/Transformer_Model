import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer.tokenization.bpe_tokenizer import BPETokenizer
from transformer.tokenization.hf_bpe_tokenizer import HuggingfaceBPETokenizer