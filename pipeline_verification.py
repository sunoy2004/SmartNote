"""
Pipeline verification script to confirm all components are working
"""
import sys
import os

# Add current directory to path
sys.path.append(".")

def verify_environment():
    """Verify that all required packages are installed"""
    try:
        import torch
        print("✓ PyTorch installed successfully")
        print(f"  Version: {torch.__version__}")
        
        import torchaudio
        print("✓ torchaudio installed successfully")
        
        import librosa
        print("✓ librosa installed successfully")
        
        import datasets
        print("✓ datasets installed successfully")
        
        import fastapi
        print("✓ FastAPI installed successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Environment verification failed: {e}")
        return False

def verify_model_architectures():
    """Verify that model architectures can be instantiated"""
    try:
        # Test ASR model
        from models.asr.asr_model import create_model
        asr_model = create_model()
        params = sum(p.numel() for p in asr_model.parameters())
        print("✓ ASR model instantiated successfully")
        print(f"  Parameters: {params:,}")
        
        # Test Summarizer model
        from models.summarizer.transformer import create_summarizer
        summarizer_model = create_summarizer(vocab_size=100)
        params = sum(p.numel() for p in summarizer_model.parameters())
        print("✓ Summarizer model instantiated successfully")
        print(f"  Parameters: {params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model architecture verification failed: {e}")
        return False

def verify_training_scripts():
    """Verify that training scripts can be imported"""
    try:
        from training.train_asr import main as asr_main
        print("✓ ASR training script imported successfully")
        
        from training.train_summarizer import main as summarizer_main
        print("✓ Summarizer training script imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Training script verification failed: {e}")
        return False

def verify_data_loading():
    """Verify that dataset loaders can be imported"""
    try:
        from training.dataset_loader import ASRDataset, SummarizationDataset
        print("✓ Dataset loaders imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loader verification failed: {e}")
        return False

def verify_backend():
    """Verify that backend components can be imported"""
    try:
        from backend.main import app
        print("✓ Backend API imported successfully")
        
        from backend.asr_handler import ASRHandler
        print("✓ ASR handler imported successfully")
        
        from backend.summarizer_handler import SummarizerHandler
        print("✓ Summarizer handler imported successfully")
        
        from backend.supabase_connector import SupabaseConnector
        print("✓ Supabase connector imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Backend verification failed: {e}")
        return False

def verify_supabase():
    """Verify that Supabase components can be imported"""
    try:
        from supabase import create_client
        print("✓ Supabase client imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Supabase verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("="*60)
    print("AuthentiX Voice-to-Notes Pipeline Verification")
    print("="*60)
    
    all_tests_passed = True
    
    print("\n1. Environment Verification:")
    all_tests_passed &= verify_environment()
    
    print("\n2. Model Architecture Verification:")
    all_tests_passed &= verify_model_architectures()
    
    print("\n3. Training Script Verification:")
    all_tests_passed &= verify_training_scripts()
    
    print("\n4. Data Loading Verification:")
    all_tests_passed &= verify_data_loading()
    
    print("\n5. Backend Verification:")
    all_tests_passed &= verify_backend()
    
    print("\n6. Supabase Verification:")
    all_tests_passed &= verify_supabase()
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("\nYour AuthentiX Voice-to-Notes pipeline is ready for use.")
        print("\nNext steps:")
        print("1. Run training: python train_models.py")
        print("2. Upload models: python supabase/upload_models.py")
        print("3. Start backend: cd backend && python main.py")
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please check the errors above and resolve them.")
    print("="*60)

if __name__ == "__main__":
    main()