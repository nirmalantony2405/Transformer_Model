import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=100):
        """
        Initialize the BPETokenizer with a vocabulary size.
        If vocab_size is not provided, it defaults to 100.
        """
        self.vocab_size = vocab_size
        self.base_vocab = []
        self.merge_rules = []

    def get_base_vocab(self, corpus):
        """
        Extract the base vocabulary from the corpus by splitting into characters.
        """
        words = [word for sentence in corpus for word in sentence.split()]
        char_vocab = set("".join(words))
        self.base_vocab = sorted(char_vocab)
        return self.base_vocab

    def train(self, corpus):
        """
        Train the tokenizer by finding the most frequent bigrams and creating merge rules.
        """
        # Preprocess the corpus
        words = [re.sub(r'[^\w\s]', '', sentence).split() for sentence in corpus]
        word_freqs = Counter()
        for word in words:
            word_freqs.update(word)
        word_freqs = {" ".join(word): freq for word, freq in word_freqs.items()}

        # Merge rules training loop
        for _ in range(self.vocab_size - len(self.base_vocab)):
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                chars = word.split()
                for i in range(len(chars) - 1):
                    pair_freqs[(chars[i], chars[i + 1])] += freq
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merge_rules.append(best_pair)

            updated_word_freqs = {}
            bigram = " ".join(best_pair)
            for word, freq in word_freqs.items():
                updated_word = word.replace(bigram, "".join(best_pair))
                updated_word_freqs[updated_word] = freq
            word_freqs = updated_word_freqs

        return self.base_vocab + ["".join(pair) for pair in self.merge_rules]

    def tokenize(self, word):
        """
        Tokenize a given word based on the trained merge rules.
        """
        for rule in self.merge_rules:
            bigram = "".join(rule)
            word = word.replace(bigram, bigram)
        return list(word)

    def save(self, output_dir):
        """
        Save the tokenizer's vocabulary and merge rules to files.
        """
        vocab_path = f"{output_dir}/vocab.json"
        merges_path = f"{output_dir}/merges.txt"

        # Save vocabulary
        vocab = {f"token_{i}": token for i, token in enumerate(self.base_vocab + ["".join(pair) for pair in self.merge_rules])}
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            import json
            json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)

        # Save merge rules
        with open(merges_path, "w", encoding="utf-8") as merges_file:
            merges_file.write("#version: 0.2\n")
            for pair in self.merge_rules:
                merges_file.write(f"{' '.join(pair)}\n")

        print(f"Tokenizer saved to {output_dir}")