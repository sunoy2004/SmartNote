"""
Summary generation utilities for the Transformer model
"""
import torch
import math

def generate_summary(model, src, max_length=100, start_token=1, end_token=2, tokenizer=None):
    """
    Generate summary using greedy decoding
    
    Args:
        model: Trained transformer model
        src: [batch, src_len] - source token IDs
        max_length: maximum summary length
        start_token: BOS token ID
        end_token: EOS token ID
        tokenizer: tokenizer instance for special token handling
    
    Returns:
        generated: [batch, gen_len] - generated token IDs
    """
    model.eval()
    batch_size = src.size(0)
    device = src.device
    
    # Encode source
    src_emb = model.encoder_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.encoder_pos(src_emb)
    
    # Create padding mask for source
    src_padding_mask = (src == 0)
    
    # Encode
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)
    
    # Initialize decoder input with start token
    tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    
    for _ in range(max_length):
        # Create causal mask
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Decode
        tgt_emb = model.decoder_embedding(tgt) * math.sqrt(model.d_model)
        tgt_emb = model.decoder_pos(tgt_emb)
        
        output = model.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Get next token logits
        logits = model.output_layer(output[:, -1, :])
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append to sequence
        tgt = torch.cat([tgt, next_token], dim=1)
        
        # Stop if all sequences have end token
        if (next_token == end_token).all():
            break
    
    return tgt