import torch
from torch.utils.data import DataLoader
from torch import nn
import os
import sys

# project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from transformer.modelling.transformer_model import TransformerModel
from transformer.schedulers.LR_sheduler import TransformerLRScheduler
from transformer.data.dataset import get_dataloaders
from torch.utils.data import DataLoader


# Training Script
def train_model():
    # Configurations
    d_model = 64
    n_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    max_len = 50
    batch_size = 8
    num_epochs = 5
    warmup_steps = 4000
    learning_rate = 1e-4
    weight_decay = 1e-4

    # Load Dataloaders and Tokenizer
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(batch_size,max_len)
    vocab_size = len(tokenizer)

    # Model, Loss, Optimizer, Scheduler
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = TransformerLRScheduler(optimizer, d_model, warmup_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            output = output.reshape(-1, vocab_size)
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = model(src, tgt_input)
                output = output.reshape(-1, vocab_size)
                tgt_output = tgt_output.reshape(-1)
                loss = criterion(output, tgt_output)

                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")


# Run Training
if __name__ == "__main__":
    train_model()

print("Training Complete")