"""
Complete pipeline runner for AuthentiX Voice-to-Notes system
"""
import subprocess
import sys
import os

def run_training():
    """Run model training"""
    print("Step 1: Training models...")
    try:
        result = subprocess.run([sys.executable, "train_models.py"], 
                              check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(e.stderr)
        return False

def upload_models():
    """Upload models to Supabase"""
    print("Step 2: Uploading models to Supabase...")
    try:
        result = subprocess.run([sys.executable, "supabase/upload_models.py"], 
                              check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Model upload failed: {e}")
        print(e.stderr)
        return False

def run_backend():
    """Run backend API server"""
    print("Step 3: Starting backend API server...")
    print("Backend server starting on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    try:
        os.chdir("backend")
        result = subprocess.run([sys.executable, "main.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Backend server failed: {e}")
        return False

def main():
    """Main pipeline function"""
    print("="*60)
    print("AuthentiX Voice-to-Notes Complete Pipeline")
    print("="*60)
    
    # Step 1: Train models
    if not run_training():
        print("Pipeline stopped due to training failure.")
        return
    
    # Step 2: Upload models
    if not upload_models():
        print("Warning: Model upload failed. Continuing with local models.")
    
    # Step 3: Run backend
    print("\n" + "="*60)
    print("Starting Backend Server")
    print("="*60)
    run_backend()

if __name__ == "__main__":
    main()