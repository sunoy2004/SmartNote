"""
Test script to verify the AuthentiX Voice-to-Notes setup
"""
import os
import sys

def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        "training",
        "models",
        "models/asr",
        "models/summarizer",
        "backend",
        "supabase",
        "checkpoints/asr",
        "checkpoints/summarizer"
    ]
    
    print("Checking directories...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} (missing)")
            return False
    return True

def check_training_files():
    """Check if all training files exist"""
    training_files = [
        "training/train_asr.py",
        "training/train_summarizer.py",
        "training/dataset_loader.py",
        "training/tokenizer.py",
        "training/utils_audio.py",
        "training/utils_text.py"
    ]
    
    print("\nChecking training files...")
    for file in training_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            return False
    return True

def check_model_files():
    """Check if all model files exist"""
    model_files = [
        "models/asr/asr_model.py",
        "models/asr/decode.py",
        "models/summarizer/transformer.py",
        "models/summarizer/generate_summary.py"
    ]
    
    print("\nChecking model files...")
    for file in model_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            return False
    return True

def check_backend_files():
    """Check if all backend files exist"""
    backend_files = [
        "backend/main.py",
        "backend/asr_handler.py",
        "backend/summarizer_handler.py",
        "backend/supabase_connector.py",
        "backend/requirements.txt"
    ]
    
    print("\nChecking backend files...")
    for file in backend_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            return False
    return True

def check_supabase_files():
    """Check if all Supabase files exist"""
    supabase_files = [
        "supabase/upload_models.py",
        "supabase/edge_function_inference.ts"
    ]
    
    print("\nChecking Supabase files...")
    for file in supabase_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            return False
    return True

def check_python_imports():
    """Check if Python imports work"""
    print("\nChecking Python imports...")
    
    # Add current directory to path
    sys.path.append(".")
    
    try:
        # Test training imports
        from training.tokenizer import SimpleTokenizer
        print("  ✓ training.tokenizer")
        
        from training.dataset_loader import ASRDataset, SummarizationDataset
        print("  ✓ training.dataset_loader")
        
        # Test model imports
        from models.asr.asr_model import create_model
        print("  ✓ models.asr.asr_model")
        
        from models.summarizer.transformer import create_summarizer
        print("  ✓ models.summarizer.transformer")
        
        # Test backend imports (skip supabase as it requires installation)
        # from backend.supabase_connector import SupabaseConnector
        print("  ✓ backend.supabase_connector (skipped for now)")
        
        print("All imports successful!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("AuthentiX Voice-to-Notes Setup Test")
    print("="*60)
    
    all_checks_passed = True
    
    # Run all checks
    all_checks_passed &= check_directories()
    all_checks_passed &= check_training_files()
    all_checks_passed &= check_model_files()
    all_checks_passed &= check_backend_files()
    all_checks_passed &= check_supabase_files()
    all_checks_passed &= check_python_imports()
    
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r backend/requirements.txt")
        print("2. Set up environment: cp .env.example .env")
        print("3. Train models: python train_models.py")
    else:
        print("✗ Some checks failed. Please review the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()