"""
Custom Transformer-based Summarization Model
Trained from scratch for transcript-to-notes conversion
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class CustomTransformerSummarizer(nn.Module):
    """
    Custom Transformer Encoder-Decoder for Summarization
    - 4 encoder layers
    - 4 decoder layers
    - 8 attention heads
    - d_model=256
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=4, 
                 num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, max_len=512):
        super(CustomTransformerSummarizer, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encodings
        self.encoder_pos = PositionalEncoding(d_model, max_len)
        self.decoder_pos = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            src: [batch, src_len] - source token IDs
            tgt: [batch, tgt_len] - target token IDs
            src_mask: source attention mask
            tgt_mask: target attention mask (causal)
            src_key_padding_mask: [batch, src_len] - padding mask
            tgt_key_padding_mask: [batch, tgt_len] - padding mask
        
        Returns:
            output: [batch, tgt_len, vocab_size] - logits
        """
        # Embeddings
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        
        # Positional encodings
        src_emb = self.encoder_pos(src_emb)
        tgt_emb = self.decoder_pos(tgt_emb)
        
        # Transformer
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Output projection
        output = self.output_layer(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

def create_summarizer(vocab_size):
    """Factory function to create summarizer"""
    return CustomTransformerSummarizer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=512
    )