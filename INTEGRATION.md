# Custom Model Integration Guide

This AI Notes app is built to integrate with your custom PyTorch ASR and NLP models.

## Current Implementation

The app currently uses:
- **Web Speech API** for voice transcription (placeholder for your custom ASR model)
- **Simple extractive summarization** (placeholder for your custom NLP model)
- **localStorage** for data persistence (no backend required)

## Integrating Your Custom Models

### Step 1: Host Your Python Models as APIs

You need to create REST API endpoints for your models. Here's a Flask example:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from model import CustomASRModel  # Your ASR model
from model_summarizer import CustomSummarizer  # Your NLP model

app = Flask(__name__)
CORS(app)

# Load models
asr_model = CustomASRModel()
asr_model.load_state_dict(torch.load('asr_weights.pth'))
asr_model.eval()

nlp_model = CustomSummarizer()
nlp_model.load_state_dict(torch.load('nlp_weights.pth'))
nlp_model.eval()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    # Process audio with your ASR model
    transcript = process_audio(audio_file, asr_model)
    return jsonify({'transcript': transcript})

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json['text']
    # Process text with your NLP model
    summary = generate_summary(text, nlp_model)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Step 2: Update Frontend Code

#### For ASR Integration

Replace Web Speech API in `src/components/VoiceRecorder.tsx`:

```typescript
// Add at top of file
const API_URL = 'https://your-api-url.com';

// Replace the startRecording/stopRecording logic with:
const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
const audioChunks = useRef<Blob[]>([]);

const startRecording = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const recorder = new MediaRecorder(stream);
  
  recorder.ondataavailable = (e) => {
    audioChunks.current.push(e.data);
  };
  
  recorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    const response = await fetch(`${API_URL}/transcribe`, {
      method: 'POST',
      body: formData,
    });
    
    const { transcript } = await response.json();
    setTranscript(transcript);
    audioChunks.current = [];
  };
  
  recorder.start();
  setMediaRecorder(recorder);
  setIsRecording(true);
};

const stopRecording = () => {
  if (mediaRecorder) {
    mediaRecorder.stop();
    setIsRecording(false);
  }
};
```

#### For NLP Integration

Replace the summarizer in `src/lib/summarizer.ts`:

```typescript
const API_URL = 'https://your-api-url.com';

export const summarizeText = async (text: string): Promise<string> => {
  const response = await fetch(`${API_URL}/summarize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  
  if (!response.ok) {
    throw new Error('Summarization failed');
  }
  
  const { summary } = await response.json();
  return summary;
};
```

### Step 3: Deploy Your API

Deploy your Python API to:
- **Heroku** (easy, free tier available)
- **AWS Lambda** (serverless)
- **Google Cloud Run** (containerized)
- **DigitalOcean** (simple VPS)
- **Railway** (modern platform)

### Step 4: Update API URL

Add your API URL to the environment or directly in the code:

```typescript
// In both files
const API_URL = import.meta.env.VITE_API_URL || 'https://your-api-url.com';
```

## Model Files Overview

Your uploaded models:
- `model.py` - CustomASRModel (CNN + Bi-LSTM architecture)
- `model_summarizer.py` - CustomSummarizer (Transformer-based)
- `train_asr.py` - ASR training script
- `train_nlp.py` - NLP training script
- `evaluate_asr.py` - ASR evaluation
- `evaluate_nlp.py` - NLP evaluation

## Testing

1. Test your API endpoints locally first
2. Use tools like Postman or curl to verify responses
3. Update the frontend code
4. Test end-to-end in the browser

## Need Help?

- The app is fully functional with browser APIs
- Your custom models will provide better accuracy
- Contact support if you need help with deployment
