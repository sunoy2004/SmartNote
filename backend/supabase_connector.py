"""
Supabase Connector for Model Management
Handles downloading models from Supabase Storage and managing metadata
"""
import os
import sys
import torch
from supabase import create_client, Client
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

class SupabaseConnector:
    """Connector for Supabase integration"""
    
    def __init__(self):
        # Get Supabase credentials from environment variables
        # Try service role key first (for server-side operations), then regular key
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # Create Supabase client
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Local cache directory
        self.cache_dir = "model_cache"
        # Don't create cache directory to avoid caching issues
        # os.makedirs(self.cache_dir, exist_ok=True)
    
    def download_model(self, model_name: str, version: str = "latest") -> str:
        """
        Download model from Supabase Storage
        
        Args:
            model_name: Name of the model file (e.g., "asr_model.pth")
            version: Model version (default: "latest")
            
        Returns:
            Path to downloaded model file
        """
        try:
            # Define local path
            local_path = os.path.join(self.cache_dir, model_name)
            
            # Always try to download from Supabase Storage first
            print(f"Downloading {model_name} from Supabase Storage...")
            try:
                # Create cache directory only when needed
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(local_path, 'wb') as f:
                    res = self.client.storage.from_("models").download(model_name)
                    f.write(res)
                
                print(f"✓ Downloaded {model_name} to {local_path}")
                return local_path
            except Exception as e:
                # Handle case where bucket doesn't exist
                if "Bucket not found" in str(e):
                    print(f"Bucket 'models' not found in Supabase Storage")
                else:
                    print(f"Error downloading from Supabase: {e}")
                # Fall back to local file if it exists
                # Correct the paths for local fallback
                local_fallback = None
                if model_name == "asr_model.pth":
                    local_fallback = os.path.join("..", "checkpoints", "asr", "best_model.pth")
                elif model_name == "summarizer_model.pth":
                    local_fallback = os.path.join("..", "checkpoints", "summarizer", "best_summarizer.pth")
                elif model_name == "tokenizer.json":
                    local_fallback = os.path.join("..", "training", "tokenizer.json")
                
                if local_fallback and os.path.exists(local_fallback):
                    print(f"Falling back to local file: {local_fallback}")
                    # Copy local file to cache with the correct name
                    import shutil
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    shutil.copy2(local_fallback, local_path)
                    print(f"Copied local file to cache: {local_path}")
                    return local_path
                else:
                    raise FileNotFoundError(f"Model {model_name} not found locally or in Supabase")
            
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            # If download fails, check if we have a local copy
            local_fallback = None
            if model_name == "asr_model.pth":
                local_fallback = os.path.join("..", "checkpoints", "asr", "best_model.pth")
            elif model_name == "summarizer_model.pth":
                local_fallback = os.path.join("..", "checkpoints", "summarizer", "best_summarizer.pth")
            elif model_name == "tokenizer.json":
                local_fallback = os.path.join("..", "training", "tokenizer.json")
            
            if local_fallback and os.path.exists(local_fallback):
                print(f"Using local version of {model_name}")
                # Create cache directory and copy local file to cache
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                import shutil
                shutil.copy2(local_fallback, local_path)
                return local_path
            else:
                raise
    
    def upload_model(self, local_path: str, model_name: str) -> bool:
        """
        Upload model to Supabase Storage
        
        Args:
            local_path: Path to local model file
            model_name: Name to store model as in Supabase
            
        Returns:
            True if successful
        """
        try:
            print(f"Uploading {model_name} to Supabase Storage...")
            with open(local_path, 'rb') as f:
                self.client.storage.from_("models").upload(
                    file=f,
                    path=model_name,
                    file_options={"content-type": "application/octet-stream"}
                )
            
            print(f"✓ Uploaded {model_name} to Supabase Storage")
            return True
            
        except Exception as e:
            print(f"Error uploading model {model_name}: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> dict:
        """
        Get model metadata from Supabase database
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata dictionary
        """
        try:
            response = self.client.table("models_meta").select("*").eq("name", model_name).execute()
            if response.data:
                return response.data[0]
            return {}
        except Exception as e:
            print(f"Error fetching metadata for {model_name}: {e}")
            return {}
    
    def update_model_metadata(self, model_name: str, metadata: dict) -> bool:
        """
        Update model metadata in Supabase database
        
        Args:
            model_name: Name of the model
            metadata: Metadata to update
            
        Returns:
            True if successful
        """
        try:
            # Check if model exists
            existing = self.get_model_metadata(model_name)
            
            if existing:
                # Update existing record
                self.client.table("models_meta").update(metadata).eq("name", model_name).execute()
            else:
                # Insert new record
                metadata["name"] = model_name
                self.client.table("models_meta").insert(metadata).execute()
            
            print(f"✓ Updated metadata for {model_name}")
            return True
            
        except Exception as e:
            print(f"Error updating metadata for {model_name}: {e}")
            return False
    
    def load_model_checkpoint(self, model_name: str, model_class, *args, **kwargs):
        """
        Load model checkpoint from Supabase with fallback to local cache
        
        Args:
            model_name: Name of the model file
            model_class: Model class to instantiate
            *args, **kwargs: Arguments for model initialization
            
        Returns:
            Tuple of (model, checkpoint_data, device)
        """
        device = torch.device("cpu")  # CPU-only for lightweight deployment
        
        try:
            # Download model
            model_path = self.download_model(model_name)
            
            # Load checkpoint with compatibility fix
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            except Exception as e1:
                # Fallback for older checkpoints or corrupted files
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                except Exception as e2:
                    print(f"Error loading checkpoint with weights_only=True: {e1}")
                    print(f"Error loading checkpoint with weights_only=False: {e2}")
                    # If both fail, try to use local checkpoint directly
                    local_fallback = os.path.join("..", "checkpoints", "asr", model_name)
                    if model_name == "summarizer_model.pth":
                        local_fallback = os.path.join("..", "checkpoints", "summarizer", model_name)
                    elif model_name == "tokenizer.json":
                        local_fallback = os.path.join("..", "training", "tokenizer.json")
                    
                    if os.path.exists(local_fallback):
                        print(f"Trying to load from local fallback: {local_fallback}")
                        try:
                            checkpoint = torch.load(local_fallback, map_location=device, weights_only=False)
                            # Copy to cache for next time
                            import shutil
                            cache_path = os.path.join(self.cache_dir, model_name)
                            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                            shutil.copy2(local_fallback, cache_path)
                        except Exception as e3:
                            print(f"Error loading from local fallback: {e3}")
                            raise
                    else:
                        raise
            
            # Create model instance
            model = model_class(*args, **kwargs)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            print(f"✓ Loaded {model_name} successfully")
            return model, checkpoint, device
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise
