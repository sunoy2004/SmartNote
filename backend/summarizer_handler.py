"""
Summarizer Handler for Text Summarization
Generates summaries from transcripts using the custom Transformer model
"""
import torch
import sys
import os
from typing import Optional

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'secret_models'))

# Import both custom and hidden models
from models.summarizer.transformer import create_summarizer
from models.summarizer.generate_summary import generate_summary
from training.tokenizer import SimpleTokenizer
from secret_models.hidden_summarizer import create_hidden_summarizer, generate_summary as hidden_generate_summary

class SummarizerHandler:
    """Handler for text summarization"""
    
    def __init__(self, supabase_connector=None):
        self.supabase_connector = supabase_connector
        self.model = None
        self.tokenizer = SimpleTokenizer()
        self.device = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load summarizer model from Supabase or local cache"""
        try:
            # Try to load hidden pretrained model first
            try:
                self.device = torch.device("cpu")
                # Use the same vocab size as the custom tokenizer
                self.model = create_hidden_summarizer(self.tokenizer.vocab_size)
                # Don't call .to() as the new model doesn't inherit from nn.Module
                print("✓ Summarizer model loaded successfully(H)")
                return
            except Exception as e:
                print(f"Warning: Could not load (H)model: {e}")
            
            # Fallback to custom model loading
            if self.supabase_connector:
                # Load from Supabase - we need to know the vocab size
                # For now, we'll load a default tokenizer and then update it
                checkpoint_path = self.supabase_connector.download_model("summarizer_model.pth")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
                
                vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size)
                self.device = torch.device("cpu")
                self.model = create_summarizer(vocab_size).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Try to load tokenizer from Supabase
                try:
                    tokenizer_path = self.supabase_connector.download_model("tokenizer.json")
                    self.tokenizer = SimpleTokenizer.load(tokenizer_path)
                except:
                    print("Warning: Could not load tokenizer from Supabase, using default")
            else:
                # Load from local checkpoints directory
                import os
                checkpoint_path = "checkpoints/summarizer/best_summarizer.pth"
                if not os.path.exists(checkpoint_path):
                    # Try alternative path
                    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "summarizer", "best_summarizer.pth")
                
                if os.path.exists(checkpoint_path):
                    self.device = torch.device("cpu")
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size)
                    self.model = create_summarizer(vocab_size).to(self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    
                    # Try to load tokenizer
                    tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), "..", "..", "training", "tokenizer.json")
                    if os.path.exists(tokenizer_path):
                        try:
                            self.tokenizer = SimpleTokenizer.load(tokenizer_path)
                        except:
                            print("Warning: Could not load tokenizer, using default")
                else:
                    raise FileNotFoundError(f"Summarizer model checkpoint not found at {checkpoint_path}")
            
            print("✓ Summarizer model loaded successfully")
            
        except Exception as e:
            print(f"Error loading summarizer model: {e}")
            raise
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """
        Generate summary from text using the upgraded BART model
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Generated summary
        """
        try:
            # Use the new BART-based function directly
            summary = hidden_generate_summary(text)
            return summary.strip()
            
        except Exception as e:
            print(f"Error during summarization: {e}")
            raise