import re
import json
import logging
from collections import defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm

class OdiaBPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.base_vocab = set()
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        self.set_odia_characters()
        #Odia word pattern to extract words from the dataset, it would exclude english characters and numbers
        self.odia_word_pattern = re.compile(r""" ?[\u0B00-\u0B7F]+[^\sA-Za-z0-9]*| [^\sA-Za-z0-9]+| \s+(?!\S)| \s+""")
        
    @classmethod
    def from_id_to_token(cls, id_to_token: Dict[int, str]):
        """Alternative constructor to initialize with a pre-existing id_to_token mapping."""
        instance = cls(vocab_size=0)  # vocab_size is irrelevant here
        instance.id_to_token = id_to_token
        instance.token_to_id = {v: k for k, v in id_to_token.items()}
        return instance

    def set_odia_characters(self):
        odia_characters = [
            # Vowels
            'ଅ', 'ଆ', 'ଇ', 'ଈ', 'ଉ', 'ଊ', 'ଋ', 'ୠ', 'ଏ', 'ଐ', 'ଓ', 'ଔ',
            # Gutturals
            'କ', 'ଖ', 'ଗ', 'ଘ', 'ଙ',
            # Palatals
            'ଚ', 'ଛ', 'ଜ', 'ଝ', 'ଞ',
            # Cerebrals
            'ଟ', 'ଠ', 'ଡ', 'ଢ', 'ଣ',
            # Dentals
            'ତ', 'ଥ', 'ଦ', 'ଧ', 'ନ',
            # Labials
            'ପ', 'ଫ', 'ବ', 'ଭ', 'ମ',
            # Semi-vowels
            'ଯ', 'ର', 'ଲ', 'ଵ', 'ଳ',
            # Sibilants
            'ଶ', 'ଷ', 'ସ',
            # Aspirate
            'ହ',
            # Numbers
            '୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯',
            # Vowel Marks
            'ି', 'ୀ', 'ୁ', 'ୂ', 'ୃ', 'ୄ', 'େ', 'ୈ', 'ୋ', 'ୌ',
            # Special Marks
            'ଁ', 'ଂ', 'ଃ', '୍',
            # Punctuation and Signs
            '।', '॥', '୰'
        ]

        self.base_vocab.update(odia_characters)
        self.base_vocab.update([    
            ' ', '\n', '\t','-'  # Whitespace characters
        ])
        

    def _get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs in the vocabulary"""
        pairs = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i + 2])] += 1
        return pairs

    def _merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of the most frequent pair"""
        first, second = pair
        new_words = []
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        
        return new_words

    def train(self, dataset: str):
        logging.info("Starting training process...")
        
        # Extract words using the regex pattern
        logging.info("Extracting words from dataset...")
        words = self.odia_word_pattern.findall(dataset)
        words = [list(word) for word in words]  # Convert each word to a list of characters
        logging.info(f"Extracted {len(words)} words from dataset.")

        # Use the base vocabulary directly
        vocab = self.base_vocab
        self.token_to_id = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.id_to_token = {idx + 1: char for idx, char in enumerate(vocab)}

        # Add special tokens to the vocabulary
        for token, id in self.special_tokens.items():
            self.token_to_id[token] = id
            self.id_to_token[id] = token

        # Perform BPE merges
        logging.info("Starting BPE merges...")
        for _ in tqdm(range(self.vocab_size - len(vocab) - len(self.special_tokens)), desc="Merging BPE Pairs"):
            pairs = self._get_stats(words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            words = self._merge_vocab(words, best_pair)

            # Add new token to the vocabulary
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.token_to_id) + 1
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token
            logging.info(f"Added new token: {new_token} with ID: {new_id}")

        # Save the tokens to a JSON file
        logging.info("Saving tokens to JSON file...")
        with open('odia_tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(self.id_to_token, f, ensure_ascii=False, indent=4)
        logging.info("Training completed and tokens saved.")

    def encode(self, text: str) -> List[int]:
        """Encode text using the vocabulary built during training"""
        words = self.odia_word_pattern.findall(text)
        tokens = []
        for word in words:
            i = 0
            while i < len(word):
                # Try to find the longest token in the vocabulary
                for j in range(len(word), i, -1):
                    subword = ''.join(word[i:j])
                    if subword in self.token_to_id:
                        print(f"Matched subword: '{subword}' -> {self.token_to_id[subword]}")
                        tokens.append(self.token_to_id[subword])
                        i = j
                        break
                else:
                    # If no subword is found, use <UNK>
                    logging.debug(f"Subword not found, using <UNK> for: '{word[i]}'")
                    tokens.append(self.special_tokens['<UNK>'])
                    i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text"""
        return ''.join(self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token)

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Read the dataset and trigger the training process
    logging.info("Reading dataset...")
    with open('monolingual4.final', 'r', encoding='utf-8') as file:
        dataset = file.read()
    logging.info("Dataset read successfully.")

    tokenizer = OdiaBPETokenizer(vocab_size=5000)
    tokenizer.train(dataset)

    # Example usage of encode and decode
    text = "ଅଁଲାଭଟା, ଓଡ଼ିଶାର କଳାହାଣ୍ଡି"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    logging.info("Original Text: %s", text)
    logging.info("Encoded Tokens: %s", encoded)
    logging.info("Decoded Text: %s", decoded)

if __name__ == "__main__":
    main()
