"""
Simple test to create and save a summarizer model checkpoint
"""
import sys
import os
import torch

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

def create_and_save_summarizer():
    """Create a simple summarizer model and save it as a checkpoint"""
    try:
        print("Creating summarizer model...")
        from models.summarizer.transformer import create_summarizer
        
        # Create model with a reasonable vocab size
        vocab_size = 1000  # Typical size for our tokenizer
        model = create_summarizer(vocab_size=vocab_size)
        print(f"✓ Summarizer model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create a minimal checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab_size': vocab_size,
            'epoch': 0,
            'val_loss': 0.0
        }
        
        # Save checkpoint
        checkpoint_dir = "checkpoints/summarizer"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_summarizer.pth")
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Model checkpoint saved to {checkpoint_path}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create and save summarizer model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("Creating and Saving Summarizer Model Checkpoint")
    print("=" * 60)
    
    success = create_and_save_summarizer()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Summarizer model checkpoint created successfully!")
        print("  You can now restart the backend API to load this model.")
    else:
        print("✗ Failed to create summarizer model checkpoint.")
    print("=" * 60)

if __name__ == "__main__":
    main()