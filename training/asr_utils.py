"""
Audio processing utilities for ASR
"""
import torch
import torchaudio
import numpy as np

class AudioProcessor:
    """Process audio files into mel spectrograms"""
    
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
    def load_audio(self, audio_path):
        """Load audio file and resample to 16kHz"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def waveform_to_mel(self, waveform):
        """Convert waveform to log mel spectrogram"""
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-9)
        
        return log_mel
    
    def process_audio(self, audio_path):
        """
        Full pipeline: load audio -> mel spectrogram
        Returns: [1, time, freq] tensor ready for model
        """
        waveform = self.load_audio(audio_path)
        log_mel = self.waveform_to_mel(waveform)
        
        # Add batch and channel dimensions: [1, 1, freq, time]
        # Note: mel_spec is [channels, freq, time]
        log_mel = log_mel.unsqueeze(0)  # [1, channels, freq, time]
        
        # Transpose to [1, channels, time, freq] for CNN
        log_mel = log_mel.transpose(2, 3)
        
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
