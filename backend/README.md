# Custom ASR + NLP Backend

Complete backend implementation with custom PyTorch models for speech-to-text and summarization.

## 📁 Project Structure

```
backend/
├── app.py                      # Flask API server
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── asr/                        # ASR (Speech-to-Text)
│   ├── model.py               # CNN + BiLSTM + CTC architecture
│   ├── dataset.py             # LibriSpeech data loader
│   ├── train_asr.py           # Training script
│   ├── ctc_decoder.py         # CTC greedy decoder
│   ├── utils_audio.py         # Audio processing utilities
│   └── checkpoints/           # Trained model weights (create this)
│
└── nlp/                        # NLP (Summarization)
    ├── model_summarizer.py    # Custom Transformer encoder-decoder
    ├── tokenizer.py           # Character-level tokenizer
    ├── dataset.py             # Dataset with auto-generated summaries
    ├── train_summarizer.py    # Training script
    └── checkpoints/           # Trained model weights (create this)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (Google Colab Recommended)

#### Train ASR Model:
```bash
# Quick test (2% of data, ~5 minutes)
python asr/train_asr.py --epochs 5 --batch_size 16 --subset 2

# Full training (100% of data, several hours on GPU)
python asr/train_asr.py --epochs 50 --batch_size 16 --subset 100
```

#### Train NLP Summarizer:
```bash
# Quick test (2% of data)
python nlp/train_summarizer.py --epochs 5 --batch_size 32 --subset 2

# Full training
python nlp/train_summarizer.py --epochs 50 --batch_size 32 --subset 100
```

**Training outputs:**
- Checkpoints saved in `asr/checkpoints/` and `nlp/checkpoints/`
- Best models: `best_model.pth` and `best_summarizer.pth`
- TensorBoard logs in `runs/`

### 3. Run API Server

```bash
python app.py
```

Server runs on `http://0.0.0.0:5000`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "asr_loaded": true,
  "nlp_loaded": true
}
```

### Transcribe Audio
```bash
POST /api/asr/transcribe
Content-Type: multipart/form-data

Body:
  audio: <audio_file.wav>
```

Response:
```json
{
  "transcript": "transcribed text here"
}
```

### Summarize Text
```bash
POST /api/nlp/summarize
Content-Type: application/json

Body:
{
  "text": "long text to summarize..."
}
```

Response:
```json
{
  "summary": "short summary here"
}
```

### Voice to Notes (Complete Pipeline)
```bash
POST /api/voice-to-notes
Content-Type: multipart/form-data

Body:
  audio: <audio_file.wav>
```

Response:
```json
{
  "transcript": "full transcription",
  "summary": "summarized notes"
}
```

## 🎯 Model Details

### ASR Model (CNN + BiLSTM + CTC)
- **Architecture:**
  - 2-layer CNN frontend (feature extraction)
  - 2-layer BiLSTM (sequence modeling)
  - Linear classifier + CTC loss
  - Greedy decoder
- **Input:** 16kHz WAV audio
- **Output:** Character-level transcript
- **Training:** LibriSpeech Clean-100
- **Metrics:** WER (Word Error Rate), CER (Character Error Rate)

### NLP Summarizer (Custom Transformer)
- **Architecture:**
  - 4 encoder layers
  - 4 decoder layers
  - 8 attention heads
  - d_model=256
  - Sinusoidal positional encodings
- **Training:** Auto-generated summaries from LibriSpeech transcripts
- **Tokenizer:** Character-level
- **Metrics:** ROUGE-1 F1 score

## 🔧 Training Tips

### For Google Colab:

1. **Upload files to Colab:**
```python
from google.colab import files
# Upload all .py files
```

2. **Install dependencies:**
```python
!pip install torch torchaudio datasets
```

3. **Train with subset for testing:**
```python
!python train_asr.py --epochs 5 --batch_size 8 --subset 2
```

4. **Download checkpoints:**
```python
from google.colab import files
files.download('asr/checkpoints/best_model.pth')
```

### Memory Tips:
- Reduce `batch_size` if out of memory
- Use `--subset 2` for quick testing
- Full training requires ~8GB GPU RAM

## 🌐 Deployment Options

### Option 1: Heroku
```bash
# Add Procfile
web: python app.py

# Deploy
heroku create my-asr-app
git push heroku main
```

### Option 2: Railway
1. Connect GitHub repo
2. Add start command: `python app.py`
3. Deploy

### Option 3: Google Cloud Run
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/asr-api
gcloud run deploy --image gcr.io/PROJECT_ID/asr-api
```

### Option 4: AWS EC2
1. Launch EC2 instance (GPU recommended)
2. Install dependencies
3. Run with `nohup python app.py &`

## 🔗 Connect Frontend

Update your Lovable frontend to point to your deployed API:

```typescript
const API_URL = 'https://your-deployed-api.com';
```

## 📊 Evaluation Results

After training, check:
- **ASR:** WER and CER on test set
- **NLP:** ROUGE scores on test set
- TensorBoard: `tensorboard --logdir=runs/`

## 🐛 Troubleshooting

**Model not loading:**
- Ensure checkpoint files exist in `asr/checkpoints/` and `nlp/checkpoints/`
- Check file paths in `app.py`

**Out of memory during training:**
- Reduce batch size: `--batch_size 8`
- Use CPU: Set `device = torch.device('cpu')`

**Poor accuracy:**
- Train longer: `--epochs 100`
- Use full dataset: `--subset 100`
- Check data quality

## 📚 Additional Resources

- LibriSpeech Dataset: https://huggingface.co/datasets/nguyenvulebinh/libris_clean_100
- PyTorch Docs: https://pytorch.org/docs/
- Flask Docs: https://flask.palletsprojects.com/

## ✅ Checklist

- [ ] Trained ASR model
- [ ] Trained NLP model
- [ ] Tested API locally
- [ ] Deployed to cloud
- [ ] Connected frontend
- [ ] End-to-end testing complete
