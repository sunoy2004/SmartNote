"""
Simple training script to train both ASR and Summarizer models
"""
import subprocess
import sys
import os

def train_asr():
    """Train ASR model"""
    print("Training ASR model...")
    try:
        result = subprocess.run([
            sys.executable, "training/train_asr.py",
            "--epochs", "20",
            "--batch_size", "8",
            "--subset", "5"
        ], check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ASR training failed: {e}")
        print(e.stderr)
        return False

def train_summarizer():
    """Train Summarizer model"""
    print("Training Summarizer model...")
    try:
        result = subprocess.run([
            sys.executable, "training/train_summarizer.py",
            "--epochs", "15",
            "--batch_size", "16",
            "--subset", "5"
        ], check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Summarizer training failed: {e}")
        print(e.stderr)
        return False

def main():
    """Main training function"""
    print("="*60)
    print("AuthentiX Voice-to-Notes Training Pipeline")
    print("="*60)
    
    # Train ASR model
    if not train_asr():
        print("ASR training failed. Exiting.")
        return
    
    # Train Summarizer model
    if not train_summarizer():
        print("Summarizer training failed. Exiting.")
        return
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("Next steps:")
    print("1. Upload models to Supabase: python supabase/upload_models.py")
    print("2. Run backend API: cd backend && python main.py")
    print("="*60)

if __name__ == "__main__":
    main()