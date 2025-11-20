# AuthentiX Voice-to-Notes System

A complete pipeline for training custom ASR and summarization models on a small subset of the LibriSpeech dataset, with full Supabase integration for model storage and inference.

## System Architecture

```
├── training/                 # Training pipeline
│   ├── train_asr.py         # ASR model training script
│   ├── train_summarizer.py  # Summarizer model training script
│   ├── dataset_loader.py    # Dataset loading and preprocessing
│   ├── tokenizer.py         # Text tokenizer
│   ├── utils_audio.py       # Audio processing utilities
│   └── utils_text.py        # Text processing utilities
│
├── models/                  # Model definitions
│   ├── asr/                 # ASR model (CNN + BiLSTM + CTC)
│   │   ├── asr_model.py     # Model architecture
│   │   └── decode.py        # CTC decoding
│   └── summarizer/          # Summarizer model (Transformer)
│       ├── transformer.py   # Model architecture
│       └── generate_summary.py # Summary generation
│
├── backend/                 # FastAPI backend
│   ├── main.py              # API endpoints
│   ├── asr_handler.py       # ASR transcription handler
│   ├── summarizer_handler.py # Summarization handler
│   ├── supabase_connector.py # Supabase integration
│   └── requirements.txt     # Python dependencies
│
├── supabase/                # Supabase integration
│   ├── upload_models.py     # Model upload script
│   └── edge_function_inference.ts # Edge function for inference
│
├── checkpoints/             # Trained model checkpoints
│   ├── asr/                 # ASR model checkpoints
│   └── summarizer/          # Summarizer model checkpoints
│
└── model_cache/             # Cached models from Supabase
```

## Models

### ASR Model (Automatic Speech Recognition)
- **Architecture**: CNN + BiLSTM + CTC
- **Features**:
  - CNN frontend for feature extraction
  - 2-layer BiLSTM encoder for sequence modeling
  - Linear classifier with CTC loss
  - Greedy CTC decoding

### Summarizer Model (Text Summarization)
- **Architecture**: Transformer Encoder-Decoder
- **Features**:
  - 4-layer encoder and decoder
  - Multi-head attention with cross-attention
  - Character-level tokenizer
  - Teacher forcing during training
  - Greedy decoding for inference

## Dataset

Uses a small subset (1-5%) of the [LibriSpeech Clean-100](https://huggingface.co/datasets/nguyenvulebinh/libris_clean_100) dataset from Hugging Face:
- Training: 80% of subset
- Validation: 10% of subset
- Testing: 10% of subset

## Requirements

- Python 3.8+
- PyTorch (CPU-only version)
- Librosa for audio processing
- Supabase account for model storage

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your Supabase credentials
   ```

3. **Train models**:
   ```bash
   python training/train_asr.py --epochs 20 --batch_size 8 --subset 5
   python training/train_summarizer.py --epochs 15 --batch_size 16 --subset 5
   ```

4. **Upload models to Supabase**:
   ```bash
   python supabase/upload_models.py
   ```

5. **Run backend API**:
   ```bash
   cd backend
   python main.py
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio file
- `POST /summarize` - Summarize text
- `POST /voice-to-notes` - Complete pipeline (audio → transcript → summary)

## Supabase Edge Functions

The system includes a Supabase Edge Function for serverless inference:
- Deploy `supabase/edge_function_inference.ts` to Supabase Edge Functions
- Provides the same endpoints as the backend API
- Runs on Deno with Supabase Edge Runtime

## Deployment

### Backend Deployment
1. Deploy to any cloud platform (Heroku, Railway, AWS, etc.)
2. Set environment variables for Supabase integration
3. Ensure model checkpoints are accessible

### Supabase Setup
1. Create a Supabase project
2. Create a Storage bucket named "models"
3. Create a database table "models_meta" with columns:
   - name (text)
   - type (text)
   - framework (text)
   - uploaded_at (timestamp)
   - description (text)
4. Upload models using the upload script

## CPU Optimization

All models and training scripts are optimized for CPU-only execution:
- Reduced model sizes
- Lightweight architectures
- Efficient data loading
- Memory optimization techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.