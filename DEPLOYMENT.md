# AuthentiX Voice-to-Notes Deployment Guide

## System Overview

This document provides instructions for deploying the complete AuthentiX Voice-to-Notes system, which includes:
1. Training custom ASR and Summarization models
2. Uploading models to Supabase
3. Running the backend API server
4. Deploying Supabase Edge Functions for inference

## Prerequisites

- Python 3.8+
- Supabase account
- Git (for version control)

## Directory Structure

```
├── training/                 # Training pipeline
│   ├── train_asr.py         # ASR model training
│   ├── train_summarizer.py  # Summarizer model training
│   ├── dataset_loader.py    # Dataset handling
│   ├── tokenizer.py         # Text tokenizer
│   ├── utils_audio.py       # Audio processing
│   └── utils_text.py        # Text processing
│
├── models/                  # Model definitions
│   ├── asr/                 # ASR model (CNN + BiLSTM + CTC)
│   └── summarizer/          # Summarizer model (Transformer)
│
├── backend/                 # FastAPI backend
│   ├── main.py              # API server
│   ├── asr_handler.py       # ASR transcription
│   ├── summarizer_handler.py # Text summarization
│   ├── supabase_connector.py # Supabase integration
│   └── requirements.txt     # Dependencies
│
├── supabase/                # Supabase integration
│   ├── upload_models.py     # Model upload script
│   └── edge_function_inference.ts # Edge function
│
├── checkpoints/             # Model checkpoints
│   ├── asr/                 # ASR model checkpoints
│   └── summarizer/          # Summarizer model checkpoints
│
└── model_cache/             # Cached models from Supabase
```

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd SmartNote

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your Supabase credentials
nano .env  # or use your preferred editor
```

Required environment variables:
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key (for Edge Functions)

### 3. Model Training

Train both models using the training scripts:

```bash
# Train ASR model (CNN + BiLSTM + CTC)
python training/train_asr.py --epochs 20 --batch_size 8 --subset 5

# Train Summarizer model (Transformer)
python training/train_summarizer.py --epochs 15 --batch_size 16 --subset 5

# Or use the combined training script
python train_models.py
```

Training parameters:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--subset`: Percentage of dataset to use (1-5%)

### 4. Model Upload to Supabase

After training, upload models to Supabase Storage:

```bash
python supabase/upload_models.py
```

This script will:
- Upload ASR model checkpoint
- Upload Summarizer model checkpoint
- Upload tokenizer configuration
- Update model metadata in the database

### 5. Running the Backend API

Start the FastAPI backend server:

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Supabase Edge Functions Deployment

Deploy the Edge Function for serverless inference:

1. Install Supabase CLI:
   ```bash
   npm install -g supabase
   ```

2. Login to Supabase:
   ```bash
   supabase login
   ```

3. Deploy the Edge Function:
   ```bash
   supabase functions deploy voice-to-notes --project-ref <your-project-ref>
   ```

## API Endpoints

### Health Check
```
GET /health
```

### Audio Transcription
```
POST /transcribe
Content-Type: multipart/form-data
Body: audio file
```

### Text Summarization
```
POST /summarize
Content-Type: application/json
Body: {"text": "text to summarize"}
```

### Voice-to-Notes (Complete Pipeline)
```
POST /voice-to-notes
Content-Type: multipart/form-data
Body: audio file
```

## Supabase Configuration

### Storage Setup
1. Create a Storage bucket named `models`
2. Make the bucket publicly accessible (for Edge Functions)

### Database Setup
Create a `models_meta` table with the following columns:
- `name` (text): Model name
- `type` (text): Model type (asr/summarizer/tokenizer)
- `framework` (text): Framework used (pytorch)
- `uploaded_at` (timestamp): Upload timestamp
- `description` (text): Model description

### Environment Variables
Set these in your Supabase project settings:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

## CPU Optimization

All models and training scripts are optimized for CPU-only execution:
- Lightweight model architectures
- Reduced batch sizes
- Efficient data loading
- Memory optimization techniques

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Audio Processing Issues**: The system uses librosa instead of torchaudio to avoid compatibility issues.

3. **Supabase Connection**: Verify your Supabase credentials in the `.env` file.

4. **Model Loading**: Check that model checkpoints exist in the `checkpoints/` directory.

### Performance Tuning

1. **Training Speed**: Reduce the dataset subset percentage for faster training
2. **Memory Usage**: Decrease batch size if running out of memory
3. **Model Accuracy**: Increase epochs or dataset subset for better accuracy

## Monitoring and Maintenance

### Model Updates
1. Retrain models with new data
2. Upload updated models to Supabase
3. Update Edge Functions if needed

### Logging
The system includes basic logging. For production deployments, consider adding:
- Structured logging
- Error tracking
- Performance monitoring

## Security Considerations

1. **API Keys**: Keep Supabase keys secure
2. **File Uploads**: Validate and sanitize all uploaded files
3. **Rate Limiting**: Implement rate limiting for production use
4. **CORS**: Configure CORS appropriately for your frontend

## Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple backend instances
2. **Database**: Use Supabase's built-in scaling features
3. **Storage**: Leverage Supabase Storage CDN
4. **Compute**: Use Supabase Edge Functions for serverless inference

## Backup and Recovery

1. **Model Backups**: Supabase Storage provides automatic backups
2. **Database Backups**: Use Supabase's backup features
3. **Local Copies**: Keep local copies of model checkpoints

## Support

For issues and questions:
1. Check the README.md for documentation
2. Review error logs
3. Consult the Supabase documentation
4. Open an issue in the repository