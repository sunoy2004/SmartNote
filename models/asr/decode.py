"""
CTC Decoder for ASR predictions
"""
import torch

def greedy_decode(log_probs, blank_id=0, vocab_reverse=None):
    """
    Greedy CTC decoder
    
    Args:
        log_probs: [batch, time, num_classes] - log probabilities from model
        blank_id: ID of blank token
        vocab_reverse: Dictionary mapping token IDs to characters
    
    Returns:
        decoded_texts: List of decoded strings
    """
    batch_size = log_probs.size(0)
    decoded_texts = []
    
    for b in range(batch_size):
        # Get most likely token at each timestep
        predictions = torch.argmax(log_probs[b], dim=1)  # [time]
        
        # Remove consecutive duplicates and blanks
        decoded_tokens = []
        prev_token = None
        
        for token_id in predictions:
            token_id = token_id.item()
            
            # Skip blanks
            if token_id == blank_id:
                prev_token = None
                continue
            
            # Skip consecutive duplicates
            if token_id == prev_token:
                continue
            
            decoded_tokens.append(token_id)
            prev_token = token_id
        
        # Convert token IDs to characters
        if vocab_reverse:
            text = ''.join([vocab_reverse.get(t, '') for t in decoded_tokens])
        else:
            text = ''.join([chr(t) for t in decoded_tokens if t > 0])
        decoded_texts.append(text)
    
    return decoded_texts