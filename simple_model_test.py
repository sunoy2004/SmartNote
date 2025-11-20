import torch
import numpy as np

def test_model_output_shapes():
    """Test if the model produces the expected output shapes"""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
        
        # Create a simple test input (simulating mel spectrogram)
        # Shape should be [batch, channels, time, freq]
        batch_size = 1
        channels = 1
        time_steps = 100  # 100 time steps
        freq_bins = 80    # 80 mel frequency bins
        
        # Create random input
        test_input = torch.randn(batch_size, channels, time_steps, freq_bins)
        print(f"Test input shape: {test_input.shape}")
        
        # Load model
        from models.asr.asr_model import create_model
        model = create_model()
        print(f"Model created successfully")
        print(f"Model input expected shape: [batch, channels, time, freq]")
        
        # Run inference
        with torch.no_grad():
            output = model(test_input)
            print(f"Model output shape: {output.shape}")
            
            # Expected output shape for CTC: [batch, time, num_classes]
            batch, time, num_classes = output.shape
            print(f"Output breakdown - batch: {batch}, time: {time}, classes: {num_classes}")
            
            # Check if output values are reasonable
            output_min = output.min().item()
            output_max = output.max().item()
            print(f"Output range: [{output_min:.2f}, {output_max:.2f}]")
            
            # Softmax to get probabilities
            probs = torch.softmax(output, dim=-1)
            print(f"Probability range: [{probs.min().item():.2f}, {probs.max().item():.2f}]")
            
            # Check if any probabilities are reasonably high
            max_probs = probs.max(dim=-1)[0]
            avg_max_prob = max_probs.mean().item()
            print(f"Average max probability per timestep: {avg_max_prob:.4f}")
            
            if avg_max_prob < 0.1:
                print("WARNING: Model outputs very low confidence predictions")
                
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_output_shapes()