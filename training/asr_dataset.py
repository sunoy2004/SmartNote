"""
LibriSpeech Dataset Loader for ASR Training
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchaudio
from model import VOCAB

class LibriSpeechDataset(Dataset):
    """LibriSpeech Clean-100 dataset for ASR"""
    
    def __init__(self, split='train', sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, subset_percent=100):
        """
        Args:
            split: 'train', 'validation', or 'test'
            sample_rate: Target sample rate (16kHz)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            subset_percent: Percentage of dataset to use (for quick testing)
        """
        self.sample_rate = sample_rate
        
        # Load dataset from HuggingFace
        print(f"Loading LibriSpeech Clean-100 {split} split...")
        dataset = load_dataset("nguyenvulebinh/libris_clean_100", split="train.clean.100")
        
        # Split into train/val/test (80/10/10)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        if split == 'train':
            self.data = dataset.select(range(train_size))
        elif split == 'validation':
            self.data = dataset.select(range(train_size, train_size + val_size))
        else:  # test
            self.data = dataset.select(range(train_size + val_size, total_size))
        
        # Use subset if specified
        if subset_percent < 100:
            subset_size = int(len(self.data) * subset_percent / 100)
            self.data = self.data.select(range(subset_size))
        
        print(f"Loaded {len(self.data)} samples for {split}")
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
    def __len__(self):
        return len(self.data)
    
    def text_to_indices(self, text):
        """Convert text to character indices"""
        text = text.lower()
        indices = []
        for char in text:
            if char in VOCAB:
                indices.append(VOCAB[char])
        return indices
    
    def __getitem__(self, idx):
        """
        Returns:
            mel_spec: [1, time, freq] - log mel spectrogram
            target: [target_length] - character indices
            input_length: scalar - length of mel_spec
            target_length: scalar - length of target
        """
        sample = self.data[idx]
        
        # Get audio array and text
        audio_array = torch.FloatTensor(sample['audio']['array'])
        text = sample['text']
        
        # Ensure mono channel
        if audio_array.dim() == 1:
            audio_array = audio_array.unsqueeze(0)
        elif audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(audio_array)
        
        # Log scale
        log_mel = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-9)
        
        # Transpose to [1, time, freq]
        log_mel = log_mel.transpose(1, 2)
        
        # Convert text to indices
        target = torch.LongTensor(self.text_to_indices(text))
        
        # Get lengths
        input_length = log_mel.size(1)
        target_length = len(target)
        
        return log_mel, target, input_length, target_length

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to max length in batch
    """
    mel_specs, targets, input_lengths, target_lengths = zip(*batch)
    
    # Pad mel spectrograms
    max_input_length = max(input_lengths)
    batch_size = len(mel_specs)
    n_mels = mel_specs[0].size(2)
    
    padded_inputs = torch.zeros(batch_size, 1, max_input_length, n_mels)
    for i, mel in enumerate(mel_specs):
        length = mel.size(1)
        padded_inputs[i, :, :length, :] = mel
    
    # Pad targets
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    # Convert lengths to tensors
    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)
    
    return padded_inputs, padded_targets, input_lengths, target_lengths

def get_dataloaders(batch_size=16, num_workers=2, subset_percent=100):
    """
    Create train, validation, and test dataloaders
    
    Args:
        batch_size: Batch size
        num_workers: Number of workers for data loading
        subset_percent: Use subset of data (2 for quick testing)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = LibriSpeechDataset('train', subset_percent=subset_percent)
    val_dataset = LibriSpeechDataset('validation', subset_percent=subset_percent)
    test_dataset = LibriSpeechDataset('test', subset_percent=subset_percent)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
