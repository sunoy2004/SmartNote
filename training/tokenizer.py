"""
Simple character-level tokenizer for summarization
"""
import json
import os

class SimpleTokenizer:
    """Character-level tokenizer with special tokens"""
    
    def __init__(self):
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        self.UNK_TOKEN = '<unk>'
        
        # Build vocabulary
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        
        # Add lowercase letters
        for char in 'abcdefghijklmnopqrstuvwxyz':
            self.vocab[char] = len(self.vocab)
        
        # Add digits
        for digit in '0123456789':
            self.vocab[digit] = len(self.vocab)
        
        # Add common punctuation and symbols
        for symbol in [' ', '.', ',', '!', '?', "'", '"', '-', ':', ';', '(', ')', '\n']:
            self.vocab[symbol] = len(self.vocab)
        
        # Reverse vocabulary
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, text, add_bos=True, add_eos=True):
        """
        Encode text to token IDs
        
        Args:
            text: string to encode
            add_bos: whether to add BOS token
            add_eos: whether to add EOS token
        
        Returns:
            List of token IDs
        """
        text = text.lower()
        tokens = []
        
        if add_bos:
            tokens.append(self.vocab[self.BOS_TOKEN])
        
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab[self.UNK_TOKEN])
        
        if add_eos:
            tokens.append(self.vocab[self.EOS_TOKEN])
        
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: whether to skip special tokens
        
        Returns:
            Decoded string
        """
        special_ids = {
            self.vocab[self.PAD_TOKEN],
            self.vocab[self.BOS_TOKEN],
            self.vocab[self.EOS_TOKEN],
            self.vocab[self.UNK_TOKEN]
        }
        
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.id_to_token:
                chars.append(self.id_to_token[token_id])
        
        return ''.join(chars)
    
    def batch_encode(self, texts, max_length=512, add_bos=True, add_eos=True):
        """
        Encode batch of texts with padding
        
        Args:
            texts: List of strings
            max_length: Maximum sequence length
            add_bos: Add BOS token
            add_eos: Add EOS token
        
        Returns:
            List of token ID lists (padded)
        """
        encoded = []
        for text in texts:
            tokens = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.vocab[self.EOS_TOKEN]]
            
            # Pad if too short
            while len(tokens) < max_length:
                tokens.append(self.vocab[self.PAD_TOKEN])
            
            encoded.append(tokens)
        
        return encoded
    
    def batch_decode(self, token_ids_list, skip_special_tokens=True):
        """
        Decode batch of token ID sequences
        
        Args:
            token_ids_list: List of token ID lists
            skip_special_tokens: Skip special tokens
        
        Returns:
            List of decoded strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in token_ids_list]
    
    def save(self, filepath):
        """Save tokenizer to file"""
        tokenizer_data = {
            'vocab': self.vocab,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'r') as f:
            tokenizer_data = json.load(f)
        
        tokenizer = cls.__new__(cls)  # Create instance without calling __init__
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.id_to_token = {int(k): v for k, v in tokenizer_data['id_to_token'].items()}
        tokenizer.vocab_size = tokenizer_data['vocab_size']
        
        # Set special tokens
        tokenizer.PAD_TOKEN = '<pad>'
        tokenizer.BOS_TOKEN = '<bos>'
        tokenizer.EOS_TOKEN = '<eos>'
        tokenizer.UNK_TOKEN = '<unk>'
        
        return tokenizer