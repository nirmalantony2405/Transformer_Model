import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer_project.tokenization.bpe_tokenizer import BPETokenizer
from transformer_project.tokenization.hf_bpe_tokenizer import HuggingfaceBPETokenizer