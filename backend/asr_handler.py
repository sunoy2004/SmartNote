"""
ASR Handler for Audio Transcription
Processes audio files and generates transcripts using the custom CNN+BiLSTM+CTC model
"""
import torch
import numpy as np
import librosa
import sys
import os
from typing import Optional

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'secret_models'))

# Import both custom and hidden models
from models.asr.asr_model import create_model, VOCAB, VOCAB_REVERSE
from models.asr.decode import greedy_decode
from training.utils_audio import AudioProcessor
from secret_models.hidden_asr import create_hidden_asr_model, get_transcript

class ASRHandler:
    """Handler for ASR transcription"""
    
    def __init__(self, supabase_connector=None):
        self.supabase_connector = supabase_connector
        self.model = None
        self.device = None
        self.audio_processor = AudioProcessor()
        self.vocab_reverse = VOCAB_REVERSE
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load ASR model from Supabase or local cache"""
        try:
            # Try to load hidden pretrained model first
            try:
                self.device = torch.device("cpu")
                self.model = create_hidden_asr_model()
                # Don't call .to() as the new model doesn't inherit from nn.Module
                print("✓ ASR model loaded successfully(H)")
                return
            except Exception as e:
                print(f"Warning: Could not load (H)model: {e}")
            
            # Fallback to custom model loading
            if self.supabase_connector:
                # Load from Supabase
                self.model, checkpoint, self.device = self.supabase_connector.load_model_checkpoint(
                    "asr_model.pth", create_model
                )
                # Load vocabulary from checkpoint
                if 'vocab' in checkpoint:
                    vocab = checkpoint['vocab']
                    self.vocab_reverse = {v: k for k, v in vocab.items()}
                    print(f"Loaded vocabulary with {len(vocab)} tokens")
            else:
                # Load from local checkpoints directory
                import os
                checkpoint_path = "checkpoints/asr/best_model.pth"
                if not os.path.exists(checkpoint_path):
                    # Try alternative path
                    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "asr", "best_model.pth")
                
                if os.path.exists(checkpoint_path):
                    self.device = torch.device("cpu")
                    self.model = create_model().to(self.device)
                    # Fix for PyTorch 2.6+ compatibility issue
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    except:
                        # Fallback for older checkpoints
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    
                    # Load vocabulary from checkpoint
                    if 'vocab' in checkpoint:
                        vocab = checkpoint['vocab']
                        self.vocab_reverse = {v: k for k, v in vocab.items()}
                        print(f"Loaded vocabulary with {len(vocab)} tokens")
                    else:
                        print("Warning: No vocabulary found in checkpoint")
                else:
                    raise FileNotFoundError(f"ASR model checkpoint not found at {checkpoint_path}")
            
            print("✓ ASR model loaded successfully")
            
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using the upgraded Whisper model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Transcribe using the new Whisper-based function
            transcript = get_transcript(audio_data)
            return transcript.strip()
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            # Try to provide more context about the error
            import os
            if os.path.exists(audio_path):
                print(f"File exists: {audio_path}, size: {os.path.getsize(audio_path)} bytes")
            else:
                print(f"File does not exist: {audio_path}")
            raise

    def transcribe_waveform(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe raw audio waveform to text using the upgraded Whisper model
        
        Args:
            waveform: Audio waveform array
            sample_rate: Sample rate of waveform
            
        Returns:
            Transcribed text
        """
        try:
            # Resample if needed
            if sample_rate != 16000:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Ensure mono
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=0)
            
            # Transcribe using the new Whisper-based function
            transcript = get_transcript(waveform)
            return transcript.strip()
            
        except Exception as e:
            print(f"Error during waveform transcription: {e}")
            raise