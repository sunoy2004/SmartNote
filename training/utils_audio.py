"""
Audio processing utilities for ASR training
Uses librosa instead of torchaudio to avoid compatibility issues
"""
import librosa
import numpy as np
import torch
import soundfile as sf

class AudioProcessor:
    """Process audio files into mel spectrograms using librosa"""
    
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def load_audio(self, audio_path):
        """Load audio file and resample to 16kHz using librosa"""
        try:
            print(f"Loading audio file: {audio_path}")
            waveform, sr = librosa.load(audio_path, sr=None)
            print(f"Loaded audio: shape={waveform.shape}, sample_rate={sr}")
            
            # Resample if needed
            if sr != self.sample_rate:
                print(f"Resampling from {sr} to {self.sample_rate}")
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            
            # Ensure mono
            if len(waveform.shape) > 1:
                print(f"Converting stereo to mono: {waveform.shape}")
                waveform = np.mean(waveform, axis=0)
            
            print(f"Final waveform shape: {waveform.shape}")
            return waveform
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Try alternative loading method
            try:
                import soundfile as sf
                waveform, sr = sf.read(audio_path)
                print(f"Loaded with soundfile: shape={waveform.shape}, sample_rate={sr}")
                
                # Convert to mono if needed
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)
                
                # Resample if needed
                if sr != self.sample_rate:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                
                return waveform
            except Exception as e2:
                print(f"Error loading with soundfile: {e2}")
                raise e

    def waveform_to_mel(self, waveform):
        """Convert waveform to log mel spectrogram using librosa"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-9)
        
        return log_mel
    
    def process_audio(self, audio_path):
        """
        Full pipeline: load audio -> mel spectrogram
        Returns: [1, 1, time, freq] tensor ready for model
        """
        waveform = self.load_audio(audio_path)
        log_mel = self.waveform_to_mel(waveform)
        
        # Convert to tensor and add batch and channel dimensions: [1, 1, time, freq]
        log_mel = torch.FloatTensor(log_mel)
        log_mel = log_mel.transpose(0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, time, freq]
        
        return log_mel

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Edit distance (Levenshtein)
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 0
    return wer * 100

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # Edit distance
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    cer = d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) if len(ref_chars) > 0 else 0
    return cer * 100