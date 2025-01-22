import unittest
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from transformer_project.tokenization.bpe_tokenizer import BPETokenizer
from transformer_project.tokenization.hf_bpe_tokenizer import HuggingfaceBPETokenizer

class TestBPETokenizers(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "Machine learning helps in understanding complex patterns.",
            "Learning machine languages can be complex yet rewarding."
        ]
        self.test_sentence = "Machine learning is a subset of artificial intelligence."

    def test_custom_bpe(self):
        bpe = BPETokenizer(vocab_size=64)
        bpe.get_base_vocab(self.corpus)
        bpe.train(self.corpus)
        tokens = bpe.tokenize(self.test_sentence)
        self.assertIsInstance(tokens, list)

    def test_hf_bpe(self):
        hf_bpe = HuggingfaceBPETokenizer(vocab_size=295)
        hf_bpe.train(self.corpus)
        tokens = hf_bpe.tokenize(self.test_sentence)
        self.assertIsInstance(tokens, list)

if __name__ == "__main__":
    unittest.main()


def test_bpe_tokenizer():
    tokenizer = BPETokenizer()
    tokenizer.train(["path_to_text_file"])
    tokenizer.save("output_path")
    print("Tokenizer test passed!")