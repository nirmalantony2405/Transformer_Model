from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import GPT2Tokenizer

class HuggingfaceBPETokenizer:
    def __init__(self, vocab_size):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.vocab_size = vocab_size

    def train(self, corpus):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
        self.tokenizer.train_from_iterator(corpus, trainer)

    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

def create_gpt2_tokenizer(vocab_path=None, merges_path=None, pretrained_model_name="gpt2"):
    """
    Create a GPT2 tokenizer.
    
    Args:
        vocab_path (str): Path to the vocabulary file.
        merges_path (str): Path to the merges file.
        pretrained_model_name (str): Pretrained model name (default: "gpt2").
        
    Returns:
        GPT2Tokenizer: Tokenizer object.
    """
    vocab_path = r"C:\Users\nirma\transformer_project\tokenizer_output\vocab.json"
    merges_path = r"C:\Users\nirma\transformer_project\tokenizer_output\merges.txt"
    
    if vocab_path and merges_path:
        tokenizer = GPT2Tokenizer(vocab_file=vocab_path, merges_file=merges_path)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)
    return tokenizer