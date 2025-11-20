import torch
import os

def check_model_checkpoint(checkpoint_path=None):
    """Check the model checkpoint for issues"""
    try:
        if checkpoint_path is None:
            checkpoint_path = os.path.join("checkpoints", "asr", "best_model.pth")
            
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
            
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Try loading with weights_only=False for compatibility with older checkpoints
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        except:
            # Fallback to weights_only=True with safe globals
            import numpy
            torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
            
        print(f"Checkpoint loaded successfully")
        
        # Print checkpoint keys
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Model state dict keys: {len(state_dict)} parameters")
            
            # Check some parameter shapes
            for name, param in list(state_dict.items())[:5]:
                print(f"  {name}: {param.shape}")
                
            # Check parameter values
            total_params = sum(param.numel() for param in state_dict.values())
            print(f"Total parameters: {total_params:,}")
            
            # Check for NaN or inf values
            has_nan = False
            has_inf = False
            for param in state_dict.values():
                if torch.isnan(param).any():
                    has_nan = True
                if torch.isinf(param).any():
                    has_inf = True
                    
            print(f"Contains NaN values: {has_nan}")
            print(f"Contains Inf values: {has_inf}")
            
            # Check value ranges
            all_values = []
            for param in state_dict.values():
                all_values.append(param.flatten())
            all_values = torch.cat(all_values)
            print(f"Parameter value range: [{all_values.min():.4f}, {all_values.max():.4f}]")
            
        else:
            print("No model_state_dict found in checkpoint")
            
        # Check other checkpoint data
        if 'vocab' in checkpoint:
            vocab = checkpoint['vocab']
            print(f"Vocabulary size: {len(vocab)}")
        else:
            print("No vocabulary found in checkpoint")
            
        if 'epoch' in checkpoint:
            print(f"Training epoch: {checkpoint['epoch']}")
        else:
            print("No epoch information found")
            
        if 'loss' in checkpoint:
            print(f"Training loss: {checkpoint['loss']}")
        else:
            print("No loss information found")
            
        # Check validation metrics if available
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']}")
        if 'val_wer' in checkpoint:
            print(f"Validation WER: {checkpoint['val_wer']}")
        if 'val_cer' in checkpoint:
            print(f"Validation CER: {checkpoint['val_cer']}")
            
    except Exception as e:
        print(f"Error checking checkpoint: {e}")
        import traceback
        traceback.print_exc()

def check_all_checkpoints():
    """Check all available checkpoints"""
    print("=" * 60)
    print("Checking all available ASR checkpoints")
    print("=" * 60)
    
    # Check main checkpoint
    main_checkpoint = os.path.join("checkpoints", "asr", "best_model.pth")
    if os.path.exists(main_checkpoint):
        print("\nMain checkpoint:")
        check_model_checkpoint(main_checkpoint)
    
    # Check 30% checkpoint
    checkpoint_30 = os.path.join("checkpoints", "asr_30percent", "best_model.pth")
    if os.path.exists(checkpoint_30):
        print("\n30% checkpoint:")
        check_model_checkpoint(checkpoint_30)
        
    # Check some epoch checkpoints
    epoch_checkpoints = [
        "checkpoint_epoch_5.pth",
        "checkpoint_epoch_10.pth", 
        "checkpoint_epoch_15.pth",
        "checkpoint_epoch_20.pth"
    ]
    
    for epoch_file in epoch_checkpoints:
        epoch_path = os.path.join("checkpoints", "asr_30percent", epoch_file)
        if os.path.exists(epoch_path):
            print(f"\n{epoch_file}:")
            check_model_checkpoint(epoch_path)

if __name__ == "__main__":
    check_all_checkpoints()