"""
Simple Supabase Model Upload Script
Uploads trained models and tokenizer to Supabase Storage
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

# Load environment variables
load_dotenv()

from supabase import create_client
from training.tokenizer import SimpleTokenizer

def upload_models():
    """Upload trained models to Supabase"""
    try:
        # Load environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # Initialize Supabase client
        print("Initializing Supabase client...")
        supabase = create_client(supabase_url, supabase_key)
        
        # Try to list buckets to check connection
        try:
            buckets = supabase.storage.list_buckets()
            print(f"Available buckets: {[bucket.name for bucket in buckets]}")
        except Exception as e:
            print(f"Warning: Could not list buckets: {e}")
        
        # Upload ASR model
        asr_model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "asr", "best_model.pth")
        if os.path.exists(asr_model_path):
            print(f"Uploading ASR model from {asr_model_path}")
            try:
                with open(asr_model_path, 'rb') as f:
                    # Try to upload to existing "models" bucket
                    result = supabase.storage.from_("models").upload(
                        file=f,
                        path="asr_model.pth",
                        file_options={"content-type": "application/octet-stream"}
                    )
                
                print("✓ ASR model uploaded successfully")
            except Exception as e:
                print(f"✗ Failed to upload ASR model: {e}")
                print("Make sure the 'models' bucket exists in your Supabase Storage")
        else:
            print(f"ASR model not found at {asr_model_path}")
        
        # Upload Summarizer model
        summarizer_model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "summarizer", "best_summarizer.pth")
        if os.path.exists(summarizer_model_path):
            print(f"Uploading Summarizer model from {summarizer_model_path}")
            try:
                with open(summarizer_model_path, 'rb') as f:
                    result = supabase.storage.from_("models").upload(
                        file=f,
                        path="summarizer_model.pth",
                        file_options={"content-type": "application/octet-stream"}
                    )
                
                print("✓ Summarizer model uploaded successfully")
            except Exception as e:
                print(f"✗ Failed to upload Summarizer model: {e}")
                print("Make sure the 'models' bucket exists in your Supabase Storage")
        else:
            print(f"Summarizer model not found at {summarizer_model_path}")
        
        # Create and upload tokenizer
        print("Creating and uploading tokenizer...")
        try:
            tokenizer = SimpleTokenizer()
            tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "training", "tokenizer.json")
            tokenizer.save(tokenizer_path)
            
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    result = supabase.storage.from_("models").upload(
                        file=f,
                        path="tokenizer.json",
                        file_options={"content-type": "application/json"}
                    )
                
                print("✓ Tokenizer uploaded successfully")
            else:
                print(f"Tokenizer not found at {tokenizer_path}")
        except Exception as e:
            print(f"✗ Failed to create/upload tokenizer: {e}")
            print("Make sure the 'models' bucket exists in your Supabase Storage")
        
        print("\nUpload process completed!")
        print("\nIf uploads failed, please:")
        print("1. Go to your Supabase dashboard at https://app.supabase.com/")
        print("2. Select your project with URL: https://ozsghnwhrmiznnbrkour.supabase.co")
        print("3. Go to Storage → Create a new bucket named 'models'")
        print("4. Make the bucket public")
        print("5. Run this script again")
        
    except Exception as e:
        print(f"Error during upload: {e}")
        raise

if __name__ == "__main__":
    upload_models()