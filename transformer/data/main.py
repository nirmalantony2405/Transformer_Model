import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer.modelling.model import Attention
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer.modelling.model import clean_dataset, PositionalEncoding
from transformer.tokenization.bpe_tokenizer import BPETokenizer
from transformer.tokenization.hf_bpe_tokenizer import create_gpt2_tokenizer 
from transformer.data.dataset import TranslationDataset

if __name__ == "__main__":
    attention = Attention(d_model=512, num_heads=8)
    
    # sequence
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    pad_mask = attention.create_padding_mask(seq)
    future_mask = attention.create_future_mask(seq_length=5)
    
    print("Padding Mask:\n", pad_mask)
    print("Future Mask:\n", future_mask)

def main():
    # Load dataset
    dataset = load_dataset("wmt17", "de-en")['train']
    
    # Clean data
    cleaned_data = clean_dataset(dataset)
    
    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(["path_to_text_file"])
    tokenizer.save("tokenizer_output")

    # Load GPT2 tokenizer
    vocab_path = "tokenizer_output/vocab.json"
    merges_path = "tokenizer_output/merges.txt"
    gpt2_tokenizer = create_gpt2_tokenizer(vocab_path, merges_path)

    # Prepare dataset
    translation_dataset = TranslationDataset(cleaned_data, gpt2_tokenizer)
    dataloader = DataLoader(translation_dataset, batch_size=32, shuffle=True)

    # Positional Encoding 
    pe = PositionalEncoding(d_model=64, max_len=100)
    print("Dataset and Positional Encoding Prepared.")

if __name__ == "__main__":
    main()
