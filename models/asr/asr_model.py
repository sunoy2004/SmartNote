"""
Custom ASR Model - CNN + BiLSTM + CTC
Trained from scratch on LibriSpeech Clean-100
"""
import torch
import torch.nn as nn

class CustomASRModel(nn.Module):
    """
    Custom ASR Architecture:
    - CNN frontend for feature extraction
    - BiLSTM encoder for sequence modeling
    - Linear classifier for character predictions
    - CTC loss for alignment-free training
    """
    def __init__(self, input_dim=80, hidden_dim=256, num_classes=29, num_layers=2):
        super(CustomASRModel, self).__init__()
        
        # CNN Frontend - Extract acoustic features
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate CNN output size
        # After 2 maxpool layers: input_dim / 4
        cnn_output_dim = (input_dim // 4) * 64
        
        # BiLSTM Encoder - Sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Linear classifier
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
        # Log softmax for CTC
        self.log_softmax = nn.LogSoftmax(dim=2)
        
        # Fast-init weights for stability
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stability"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, time, freq] - mel spectrogram
        Returns:
            log_probs: [batch, time, num_classes] - log probabilities
        """
        # CNN feature extraction
        x = self.cnn(x)  # [batch, 64, time/4, freq/4]
        
        # Reshape for LSTM: [batch, time, features]
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3)  # [batch, time, channels, freq]
        x = x.reshape(batch, time, channels * freq)
        
        # BiLSTM encoding
        x, _ = self.lstm(x)  # [batch, time, hidden*2]
        
        # Classification
        x = self.classifier(x)  # [batch, time, num_classes]
        
        # Log softmax for CTC
        log_probs = self.log_softmax(x)
        
        return log_probs

# Character vocabulary
VOCAB = {
    '<blank>': 0,
    'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
    'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
    'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22,
    'w': 23, 'x': 24, 'y': 25, 'z': 26, ' ': 27, "'": 28
}

VOCAB_REVERSE = {v: k for k, v in VOCAB.items()}

def create_model():
    """Factory function to create model instance"""
    # Fast-init weights for stability
    return CustomASRModel(
        input_dim=80,
        hidden_dim=256,
        num_classes=len(VOCAB),
        num_layers=2
    )