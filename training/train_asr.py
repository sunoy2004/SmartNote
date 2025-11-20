"""
Training Script for Custom ASR Model (CNN + BiLSTM + CTC)
Trains on a small subset of LibriSpeech dataset
Designed for CPU-only training
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.asr.asr_model import create_model, VOCAB, VOCAB_REVERSE
from training.dataset_loader import get_asr_dataloaders
from models.asr.decode import greedy_decode
from training.utils_audio import calculate_wer, calculate_cer

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(pbar):
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        log_probs = model(inputs)  # [batch, time, num_classes]
        
        # Transpose for CTC: [time, batch, num_classes]
        log_probs = log_probs.transpose(0, 1)
        
        # Calculate loss
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_wer = 0
    total_cer = 0
    num_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for inputs, targets, input_lengths, target_lengths in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            log_probs = model(inputs)
            
            # Calculate loss
            log_probs_t = log_probs.transpose(0, 1)
            loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode predictions for metrics
            predictions = greedy_decode(log_probs, vocab_reverse=VOCAB_REVERSE)
            
            # Convert targets to text
            for i in range(len(predictions)):
                pred_text = predictions[i]
                target_text = ''.join([VOCAB_REVERSE.get(t.item(), '') for t in targets[i] if t.item() != 0])
                
                wer = calculate_wer(target_text, pred_text)
                cer = calculate_cer(target_text, pred_text)
                
                total_wer += wer
                total_cer += cer
                num_samples += 1
    
    avg_loss = total_loss / len(val_loader)
    avg_wer = total_wer / num_samples if num_samples > 0 else 0
    avg_cer = total_cer / num_samples if num_samples > 0 else 0
    
    return avg_loss, avg_wer, avg_cer

def main():
    parser = argparse.ArgumentParser(description='Train Custom ASR Model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--subset', type=int, default=5, help='Dataset subset percent (1-5 for lightweight training)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/asr', help='Checkpoint directory')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device - CPU only for lightweight training
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Model
    print("Creating model...")
    model = create_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_asr_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,  # No workers for CPU training
        subset_percent=args.subset
    )
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_wer, val_cer = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val WER: {val_wer:.2f}%")
        print(f"  Val CER: {val_cer:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_wer': val_wer,
                'val_cer': val_cer,
                'vocab': VOCAB
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
    
    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    test_loss, test_wer, test_cer = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test WER: {test_wer:.2f}%")
    print(f"Test CER: {test_cer:.2f}%")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()