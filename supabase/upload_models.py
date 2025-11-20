"""
Supabase Model Upload Script
Uploads trained models and tokenizer to Supabase Storage
"""
import os
import sys
import torch
from dotenv import load_dotenv

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

# Load environment variables before importing SupabaseConnector
load_dotenv()

from backend.supabase_connector import SupabaseConnector
from training.tokenizer import SimpleTokenizer

def upload_models():
    """Upload trained models to Supabase"""
    try:
        # Load environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        # Use service role key for uploads (has full permissions)
        supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables")
        
        # Initialize Supabase connector with service key for uploads
        print("Initializing Supabase connector with service key...")
        # Create a custom connector for uploads
        from supabase import create_client
        supabase = create_client(supabase_url, supabase_service_key)
        
        # Upload ASR model
        asr_model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "asr", "best_model.pth")
        if os.path.exists(asr_model_path):
            print(f"Uploading ASR model from {asr_model_path}")
            try:
                with open(asr_model_path, 'rb') as f:
                    supabase.storage.from_("models").upload(
                        file=f,
                        path="asr_model.pth",
                        file_options={"content-type": "application/octet-stream"}
                    )
                
                # Update metadata
                asr_metadata = {
                    "type": "asr",
                    "framework": "pytorch",
                    "description": "Custom CNN+BiLSTM+CTC ASR model"
                }
                # Try to update metadata
                try:
                    supabase.table("models_meta").upsert({
                        "name": "asr_model.pth",
                        **asr_metadata
                    }).execute()
                except:
                    print("Warning: Could not update metadata")
                
                print("✓ ASR model uploaded successfully")
            except Exception as e:
                print(f"✗ Failed to upload ASR model: {e}")
        else:
            print(f"ASR model not found at {asr_model_path}")
        
        # Upload Summarizer model
        summarizer_model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "summarizer", "best_summarizer.pth")
        if os.path.exists(summarizer_model_path):
            print(f"Uploading Summarizer model from {summarizer_model_path}")
            try:
                with open(summarizer_model_path, 'rb') as f:
                    supabase.storage.from_("models").upload(
                        file=f,
                        path="summarizer_model.pth",
                        file_options={"content-type": "application/octet-stream"}
                    )
                
                # Update metadata
                summarizer_metadata = {
                    "type": "summarizer",
                    "framework": "pytorch",
                    "description": "Custom Transformer Encoder-Decoder Summarizer model"
                }
                try:
                    supabase.table("models_meta").upsert({
                        "name": "summarizer_model.pth",
                        **summarizer_metadata
                    }).execute()
                except:
                    print("Warning: Could not update metadata")
                
                print("✓ Summarizer model uploaded successfully")
            except Exception as e:
                print(f"✗ Failed to upload Summarizer model: {e}")
        else:
            print(f"Summarizer model not found at {summarizer_model_path}")
        
        # Upload tokenizer
        tokenizer = SimpleTokenizer()
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "training", "tokenizer.json")
        tokenizer.save(tokenizer_path)
        
        if os.path.exists(tokenizer_path):
            print(f"Uploading tokenizer from {tokenizer_path}")
            try:
                with open(tokenizer_path, 'rb') as f:
                    supabase.storage.from_("models").upload(
                        file=f,
                        path="tokenizer.json",
                        file_options={"content-type": "application/json"}
                    )
                
                # Update metadata
                tokenizer_metadata = {
                    "type": "tokenizer",
                    "format": "json",
                    "description": "Character-level tokenizer for summarization"
                }
                try:
                    supabase.table("models_meta").upsert({
                        "name": "tokenizer.json",
                        **tokenizer_metadata
                    }).execute()
                except:
                    print("Warning: Could not update metadata")
                
                print("✓ Tokenizer uploaded successfully")
            except Exception as e:
                print(f"✗ Failed to upload tokenizer: {e}")
        else:
            print(f"Tokenizer not found at {tokenizer_path}")
        
        print("\nUpload process completed!")
        
    except Exception as e:
        print(f"Error during upload: {e}")
        raise

if __name__ == "__main__":
    upload_models()