"""
Dataset for Summarization Training
Uses LibriSpeech transcripts and generates simple summaries
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re
from tokenizer import SimpleTokenizer

class SummarizationDataset(Dataset):
    """
    Dataset that creates summaries from LibriSpeech transcripts
    Since LibriSpeech doesn't have summaries, we auto-generate them
    """
    
    def __init__(self, split='train', max_src_len=512, max_tgt_len=128, subset_percent=100):
        """
        Args:
            split: 'train', 'validation', or 'test'
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
            subset_percent: Percentage of dataset to use
        """
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = SimpleTokenizer()
        
        # Load LibriSpeech dataset
        print(f"Loading LibriSpeech for summarization {split} split...")
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

def get_dataloaders(batch_size=16, num_workers=2, subset_percent=100):
    """
    Create dataloaders for summarization
    
    Args:
        batch_size: Batch size
        num_workers: Number of workers
        subset_percent: Dataset subset (2 for quick testing)
    
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    tokenizer = SimpleTokenizer()
    
    return train_loader, val_loader, test_loader, tokenizer
