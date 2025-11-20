"""
Dataset loader for both ASR and Summarization training
Handles HuggingFace dataset loading with librosa/soundfile for audio
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import librosa
import numpy as np
import re
from training.tokenizer import SimpleTokenizer

class ASRDataset(Dataset):
    """ASR Dataset using librosa for audio processing"""
    
    def __init__(self, split='train', sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, subset_percent=5):
        """
        Args:
            split: 'train', 'validation', or 'test'
            sample_rate: Target sample rate (16kHz)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            subset_percent: Percentage of dataset to use (1-5% for lightweight training)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Character vocabulary for ASR
        self.vocab = {
            '<blank>': 0,
            'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
            'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
            'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22,
            'w': 23, 'x': 24, 'y': 25, 'z': 26, ' ': 27, "'": 28
        }
        
        # Load dataset from HuggingFace - using only a small subset
        print(f"Loading LibriSpeech Clean-100 {split} split...")
        # Use the correct syntax for dataset loading
        if subset_percent < 100:
            # For very small subsets, use a minimum of 1%
            actual_percent = max(1, subset_percent)
            dataset = load_dataset("nguyenvulebinh/libris_clean_100", split=f"train.clean.100[:{actual_percent}%]")
        else:
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
        
        print(f"Loaded {len(self.data)} samples for {split}")
    
    def text_to_indices(self, text):
        """Convert text to character indices"""
        text = text.lower()
        indices = []
        for char in text:
            if char in self.vocab:
                indices.append(self.vocab[char])
        return indices
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            mel_spec: [1, time, freq] - log mel spectrogram
            target: [target_length] - character indices
            input_length: scalar - length of model output (for CTC loss)
            target_length: scalar - length of target
        """
        sample = self.data[idx]
        
        # Get audio data and text
        audio_data = sample['audio']
        text = sample['text']
        
        # Handle audio data correctly
        import numpy as np
        if isinstance(audio_data, dict):
            # Extract audio array from dict
            if 'array' in audio_data:
                audio_array = np.array(audio_data['array'])
            elif 'speech' in audio_data:
                # Some datasets use 'speech' instead of 'array'
                audio_array = np.array(audio_data['speech'])
            elif 'bytes' in audio_data and audio_data['bytes'] is not None:
                # Load audio from bytes
                import soundfile as sf
                import io
                audio_bytes = audio_data['bytes']
                audio_array, _ = sf.read(io.BytesIO(audio_bytes))
            elif 'path' in audio_data and audio_data['path'] is not None:
                # Load audio from file path
                import librosa
                audio_array, _ = librosa.load(audio_data['path'], sr=None)
            else:
                # Try to find any array-like value
                audio_array = None
                for key, value in audio_data.items():
                    # Check if value is array-like and not None
                    if value is not None and hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                        if hasattr(value, 'shape') or isinstance(value, (list, tuple, np.ndarray)):
                            audio_array = np.array(value)
                            break
                if audio_array is None:
                    # If we still can't find audio data, raise an informative error
                    raise ValueError(f"Could not extract audio array from audio data. Keys: {list(audio_data.keys())}")
        else:
            # Audio data is already an array
            audio_array = np.array(audio_data)
        
        # Convert to floating-point if needed
        if audio_array.dtype != np.float32 and audio_array.dtype != np.float64:
            # If integer type, normalize to [-1, 1] range
            if np.issubdtype(audio_array.dtype, np.integer):
                # Get the maximum value for the integer type
                max_val = np.iinfo(audio_array.dtype).max
                audio_array = audio_array.astype(np.float32) / max_val
            else:
                audio_array = audio_array.astype(np.float32)
        
        # Get sampling rate
        if isinstance(audio_data, dict) and 'sampling_rate' in audio_data:
            original_sr = audio_data['sampling_rate']
        else:
            # Default to 16kHz if sampling rate not provided
            original_sr = 16000
            
        # Resample if needed using librosa
        if original_sr != self.sample_rate:
            import librosa
            audio_array = librosa.resample(audio_array, 
                                         orig_sr=original_sr, 
                                         target_sr=self.sample_rate)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=0)
        
        # Compute mel spectrogram using librosa
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
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
        
        # Convert to tensor and transpose to [1, time, freq]
        import torch
        log_mel = torch.FloatTensor(log_mel)
        log_mel = log_mel.transpose(0, 1).unsqueeze(0)  # [1, time, freq]
        
        # Convert text to indices
        target = torch.LongTensor(self.text_to_indices(text))
        
        # Get lengths - IMPORTANT: Return the time dimension AFTER model processing
        # The model reduces time dimension by factor of 4 (2 maxpool layers with kernel=2, stride=2)
        input_time_dim = log_mel.size(1)  # Original time dimension
        output_time_dim = input_time_dim // 4  # After 2 maxpool layers
        input_length = output_time_dim  # This is what CTC loss expects
        target_length = len(target)
        
        return log_mel, target, input_length, target_length

class SummarizationDataset(Dataset):
    """
    Dataset that creates summaries from LibriSpeech transcripts
    Since LibriSpeech doesn't have summaries, we auto-generate them
    """
    
    def __init__(self, split='train', max_src_len=512, max_tgt_len=128, subset_percent=5):
        """
        Args:
            split: 'train', 'validation', or 'test'
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            subset_percent: Percentage of dataset to use (1-5% for lightweight training)
        """
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = SimpleTokenizer()
        
        # Load LibriSpeech dataset - using only a small subset
        print(f"Loading LibriSpeech for summarization {split} split...")
        dataset = load_dataset("nguyenvulebinh/libris_clean_100", split=f"train.clean.100[:{subset_percent}%]")
        
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
        
        print(f"Loaded {len(self.data)} samples for {split}")
    
    def generate_summary(self, text):
        """
        Generate a simple summary from text
        Strategy:
        1. Extract first sentence
        2. Extract sentences with key verbs/nouns
        3. Compress by removing filler words
        """
        text = text.lower().strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not sentences:
            return text[:50]
        
        # Take first sentence
        summary_parts = [sentences[0]]
        
        # Extract key sentences (with important words)
        keywords = ['said', 'told', 'asked', 'went', 'came', 'found', 'made', 'took', 'gave']
        for sentence in sentences[1:]:
            if any(keyword in sentence for keyword in keywords):
                summary_parts.append(sentence)
                if len(summary_parts) >= 3:
                    break
        
        # Combine and compress
        summary = ' '.join(summary_parts)
        
        # Remove filler words
        fillers = ['very', 'really', 'just', 'quite', 'perhaps', 'maybe']
        for filler in fillers:
            summary = summary.replace(f' {filler} ', ' ')
        
        # Truncate if too long
        if len(summary) > 200:
            summary = summary[:200].rsplit(' ', 1)[0] + '.'
        
        return summary
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            src_ids: [max_src_len] - source token IDs
            tgt_ids: [max_tgt_len] - target token IDs (input to decoder)
            tgt_labels: [max_tgt_len] - target labels (shifted by 1)
        """
        sample = self.data[idx]
        text = sample['text']
        
        # Generate summary
        summary = self.generate_summary(text)
        
        # Encode source and target
        src_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        tgt_ids = self.tokenizer.encode(summary, add_bos=True, add_eos=True)
        
        # Truncate if needed
        if len(src_ids) > self.max_src_len:
            src_ids = src_ids[:self.max_src_len-1] + [self.tokenizer.vocab[self.tokenizer.EOS_TOKEN]]
        
        if len(tgt_ids) > self.max_tgt_len:
            tgt_ids = tgt_ids[:self.max_tgt_len-1] + [self.tokenizer.vocab[self.tokenizer.EOS_TOKEN]]
        
        # Pad source
        src_padded = src_ids + [self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]] * (self.max_src_len - len(src_ids))
        
        # Pad target (input and labels)
        tgt_input = tgt_ids[:-1]  # Remove EOS for input
        tgt_label = tgt_ids[1:]   # Remove BOS for labels
        
        tgt_input_padded = tgt_input + [self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]] * (self.max_tgt_len - len(tgt_input))
        tgt_label_padded = tgt_label + [self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]] * (self.max_tgt_len - len(tgt_label))
        
        return (
            torch.LongTensor(src_padded),
            torch.LongTensor(tgt_input_padded),
            torch.LongTensor(tgt_label_padded)
        )

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to max length in batch
    """
    mel_specs, targets, input_lengths, target_lengths = zip(*batch)
    
    # Pad mel spectrograms - use the actual tensor dimensions
    max_input_length = max([mel.size(1) for mel in mel_specs])  # Use actual tensor dimensions
    batch_size = len(mel_specs)
    n_mels = mel_specs[0].size(2)
    
    padded_inputs = torch.zeros(batch_size, 1, max_input_length, n_mels)
    for i, mel in enumerate(mel_specs):
        length = mel.size(1)  # Actual length of this mel spec
        padded_inputs[i, :, :length, :] = mel
    
    # Pad targets
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    # Convert lengths to tensors - IMPORTANT: Use the lengths for CTC loss (after model processing)
    input_lengths = torch.LongTensor(input_lengths)  # These are already the reduced lengths
    target_lengths = torch.LongTensor(target_lengths)
    
    return padded_inputs, padded_targets, input_lengths, target_lengths

def get_asr_dataloaders(batch_size=8, num_workers=0, subset_percent=5):
    """
    Create train, validation, and test dataloaders for ASR
    
    Args:
        batch_size: Batch size (smaller for CPU training)
        num_workers: Number of workers for data loading
        subset_percent: Use subset of data (1-5% for lightweight training)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = ASRDataset('train', subset_percent=subset_percent)
    val_dataset = ASRDataset('validation', subset_percent=subset_percent)
    test_dataset = ASRDataset('test', subset_percent=subset_percent)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # Disable for CPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

def get_summarization_dataloaders(batch_size=16, num_workers=0, subset_percent=5):
    """
    Create dataloaders for summarization
    
    Args:
        batch_size: Batch size (smaller for CPU training)
        num_workers: Number of workers
        subset_percent: Dataset subset (1-5% for lightweight training)
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    train_dataset = SummarizationDataset('train', subset_percent=subset_percent)
    val_dataset = SummarizationDataset('validation', subset_percent=subset_percent)
    test_dataset = SummarizationDataset('test', subset_percent=subset_percent)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable for CPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    tokenizer = SimpleTokenizer()
    
    return train_loader, val_loader, test_loader, tokenizer