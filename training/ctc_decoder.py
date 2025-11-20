"""
CTC Decoder for ASR predictions
"""
import torch
from model import VOCAB_REVERSE

def greedy_decode(log_probs, blank_id=0):
    """
    Greedy CTC decoder
    
    Args:
        log_probs: [batch, time, num_classes] - log probabilities from model
        blank_id: ID of blank token
    
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
        text = ''.join([VOCAB_REVERSE.get(t, '') for t in decoded_tokens])
        decoded_texts.append(text)
    
    return decoded_texts

def decode_predictions(model_output):
    """
    Convenience function to decode model output
    
    Args:
        model_output: Output from CustomASRModel
    
    Returns:
        List of decoded strings
    """
    return greedy_decode(model_output)
