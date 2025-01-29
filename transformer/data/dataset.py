import re
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer.tokenization.bpe_tokenizer import BPETokenizer

# Cleaning Function
def clean_text(text):
    """
    Clean text by removing unwanted characters and formatting.
    """
    WHITELIST = "abcdefghijklmnopqrstuvwxyzÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = "".join([char for char in text if char in WHITELIST])
    return text.lower()


# Custom Dataset Class
class TranslationDataset(torch.utils.data.Dataset):
    """
    A custom dataset class to handle WMT17 translation data and preprocess it.
    """
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]["translation"]
        src_text = clean_text(example["de"])
        tgt_text = clean_text(example["en"])

        src_tokens = self.tokenizer.encode(
            src_text, max_length=self.max_len, padding="max_length", truncation=True
        )
        tgt_tokens = self.tokenizer.encode(
            tgt_text, max_length=self.max_len, padding="max_length", truncation=True
        )

        return {"src": torch.tensor(src_tokens, dtype=torch.long),
                "tgt": torch.tensor(tgt_tokens, dtype=torch.long)}


# Collate Function
def collate_fn(batch):
    """
    Stack tensors from the dataset into a batch.
    """
    src = torch.stack([item["src"] for item in batch], dim=0)
    tgt = torch.stack([item["tgt"] for item in batch], dim=0)
    return {"src": src, "tgt": tgt}


# Dataset Preparation Function
def get_dataloaders(batch_size, max_len, num_samples=10000):
    """
    Load a subset of the WMT17 dataset, preprocess, and return train, validation and test dataloaders.
    """
    # Load WMT17 Dataset
    dataset = load_dataset("wmt17", "de-en")

    # Reduce dataset size for faster preprocessing
    train_subset = dataset["train"].select(range(num_samples))
    valid_subset = dataset["validation"].select(range(num_samples // 10)) 
    test_subset = dataset["test"].select(range(num_samples // 10))

    # Load tokenizer
    tokenizer = BPETokenizer()

    # Wrap dataset in TranslationDataset
    train_dataset = TranslationDataset(train_subset, tokenizer, max_len)
    valid_dataset = TranslationDataset(valid_subset, tokenizer, max_len)
    test_dataset = TranslationDataset(test_subset, tokenizer, max_len)

    # DataLoaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, val_dataloader,test_dataloader,tokenizer


# Main Training Script
if __name__ == "__main__":
    batch_size = 32
    max_len = 50
    train_dataloader, val_dataloader, test_dataloader, tokenizer = get_dataloaders(batch_size, max_len)

    # DataLoaders
    for batch in train_dataloader:
        print(f"Source Batch Shape: {batch['src'].shape}")
        print(f"Target Batch Shape: {batch['tgt'].shape}")
        break
    for batch in val_dataloader:
        print(f"Source Batch Shape (Validation): {batch['src'].shape}")
        print(f"Target Batch Shape (Validation): {batch['tgt'].shape}")
        break
    for batch in test_dataloader:
        print(f"Source Batch Shape (Test): {batch['src'].shape}")
        print(f"Target Batch Shape (Test): {batch['tgt'].shape}")
        break