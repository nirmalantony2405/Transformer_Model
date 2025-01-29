from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os
import sys
from evaluate import load
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer.modelling.transformer_model import TransformerModel
from transformer.tokenization.bpe_tokenizer import BPETokenizer

# Load the WMT17 German-English dataset
dataset = load_dataset("wmt17", "de-en", split="test")

# Subset for faster experimentation
subset_size = 50
test_set = dataset.select(range(subset_size))

# Load pre-trained tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenize the source sentences
source_sentences = test_set["translation"]
tokenized_src = [
    tokenizer.encode(s["de"], return_tensors="pt", add_special_tokens=True)
    for s in source_sentences
]

# Tokenize the reference (English) sentences for BLEU evaluation
reference_sentences = [s["en"] for s in source_sentences]

# Define special tokens
start_token = tokenizer.cls_token_id or tokenizer.convert_tokens_to_ids("<sos>")
end_token = tokenizer.sep_token_id or tokenizer.convert_tokens_to_ids("<eos>")
max_len = 50 

# move the model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(
    vocab_size=len(tokenizer),
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    max_len=50,
).to(device)

checkpoint = torch.load("transformer_model.pth")
print(checkpoint.keys())

# Set the model to evaluation mode
model.eval()

# Generate translations
translations = []
with torch.no_grad():  
    for src in tokenized_src:
        src = src.to(device)  
        src_mask = torch.zeros((src.size(1), src.size(1)), device=device).bool()

        # Generate translation using greedy decoding
        try:
            generated_ids = model.greedy_decode(src, src_mask, max_len, start_token, end_token)
            translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            translations.append(translation)
        except Exception as e:
            translations.append("Error in translation")


# Load the BLEU evaluation metric
bleu = load("bleu")

# Prepare the generated translations and references for BLEU scoring
predictions = translations
references = [[ref] for ref in reference_sentences]

# Compute BLEU score
results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU Score: {results['bleu']:.4f}")

# Print some translations
for i in range(min(5, len(source_sentences))):
    print(f"Source: {source_sentences[i]['de']}")
    print(f"Generated: {translations[i]}")
    print(f"Reference: {reference_sentences[i]}")
    print("-" * 50)
