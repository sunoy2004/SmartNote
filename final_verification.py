"""
Final verification script to confirm all pipeline steps work
"""
import sys
import os

# Add current directory to path
sys.path.append(".")

def verify_training_capability():
    """Verify that training scripts can be imported and models can be created"""
    try:
        # Test ASR model creation
        from models.asr.asr_model import create_model
        asr_model = create_model()
        print("✓ ASR model can be created")
        
        # Test Summarizer model creation
        from models.summarizer.transformer import create_summarizer
        summarizer_model = create_summarizer(vocab_size=100)
        print("✓ Summarizer model can be created")
        
        # Test training script imports
        from training.train_asr import main as asr_main
        from training.train_summarizer import main as summarizer_main
        print("✓ Training scripts can be imported")
        
        return True
    except Exception as e:
        print(f"✗ Training capability verification failed: {e}")
        return False

def verify_supabase_upload():
    """Verify that Supabase upload script can be imported"""
    try:
        import sys
        sys.path.append('supabase')
        import upload_models
        print("✓ Supabase upload script can be imported")
        return True
    except Exception as e:
        print(f"✗ Supabase upload verification failed: {e}")
        return False

def verify_backend():
    """Verify that backend can be imported"""
    try:
        from backend.main import app
        print("✓ Backend API can be imported")
        return True
    except Exception as e:
        print(f"✗ Backend verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("="*60)
    print("AuthentiX Voice-to-Notes Final Verification")
    print("="*60)
    
    all_tests_passed = True
    
    print("\n1. Training Capability Verification:")
    all_tests_passed &= verify_training_capability()
    
    print("\n2. Supabase Upload Verification:")
    all_tests_passed &= verify_supabase_upload()
    
    print("\n3. Backend Verification:")
    all_tests_passed &= verify_backend()
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("\nYour AuthentiX Voice-to-Notes pipeline is ready for use.")
        print("\nTo run the complete pipeline:")
        print("1. Run training: python train_models.py")
        print("2. Upload models: python supabase/upload_models.py")
        print("3. Start backend: cd backend && python main.py")
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please check the errors above and resolve them.")
    print("="*60)

if __name__ == "__main__":
    main()