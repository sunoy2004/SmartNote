"""
Training Script for Custom Summarization Model (Transformer Encoder-Decoder)
Trains on a small subset of LibriSpeech dataset
Designed for CPU-only training
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.summarizer.transformer import create_summarizer
from training.dataset_loader import get_summarization_dataloaders
from training.utils_text import calculate_rouge_simple

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for batch_idx, (src, tgt_input, tgt_label) in enumerate(pbar):
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_label = tgt_label.to(device)
        
        # Create masks
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt_input == 0)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # Forward pass
        output = model(src, tgt_input, 
                      tgt_mask=tgt_mask,
                      src_key_padding_mask=src_padding_mask,
                      tgt_key_padding_mask=tgt_padding_mask)
        
        # Calculate loss (ignore padding)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_label.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion, tokenizer, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    rouge_scores = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for src, tgt_input, tgt_label in pbar:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_label = tgt_label.to(device)
            
            # Create masks
            src_padding_mask = (src == 0)
            tgt_padding_mask = (tgt_input == 0)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            # Forward pass for loss
            output = model(src, tgt_input,
                          tgt_mask=tgt_mask,
                          src_key_padding_mask=src_padding_mask,
                          tgt_key_padding_mask=tgt_padding_mask)
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_label.reshape(-1))
            total_loss += loss.item()
            
            # Generate summaries for ROUGE (sample a few for speed)
            if len(rouge_scores) < 100:  # Limit for efficiency
                from models.summarizer.generate_summary import generate_summary
                generated = generate_summary(model, src, max_length=100, 
                                          start_token=tokenizer.vocab[tokenizer.BOS_TOKEN],
                                          end_token=tokenizer.vocab[tokenizer.EOS_TOKEN])
                
                # Decode for metrics
                for i in range(min(len(generated), 4)):  # Sample a few for speed
                    pred_text = tokenizer.decode(generated[i].cpu().tolist())
                    ref_text = tokenizer.decode(tgt_label[i].cpu().tolist())
                    
                    rouge = calculate_rouge_simple(ref_text, pred_text)
                    rouge_scores.append(rouge)
    
    avg_loss = total_loss / len(val_loader)
    avg_rouge = np.mean(rouge_scores) if rouge_scores else 0
    
    return avg_loss, avg_rouge

def main():
    parser = argparse.ArgumentParser(description='Train Custom Summarization Model')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--subset', type=int, default=5, help='Dataset subset percent (1-5 for lightweight training)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/summarizer', help='Checkpoint directory')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device - CPU only for lightweight training
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = get_summarization_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,  # No workers for CPU training
        subset_percent=args.subset
    )
    
    # Model
    print("Creating model...")
    vocab_size = tokenizer.vocab_size
    model = create_summarizer(vocab_size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
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
        val_loss, val_rouge = validate(model, val_loader, criterion, tokenizer, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val ROUGE-1: {val_rouge:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_summarizer.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_rouge': val_rouge,
                'vocab_size': vocab_size
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
    test_loss, test_rouge = validate(model, test_loader, criterion, tokenizer, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test ROUGE-1: {test_rouge:.2f}")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()