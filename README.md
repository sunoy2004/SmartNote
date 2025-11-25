# SmartNote - Voice to Notes Application

Transform your spoken words into organized notes with AI-powered transcription and summarization.

![SmartNote Demo]((https://smart-note-assistant.netlify.app/))

## 🎯 What is SmartNote?

SmartNote is a full-stack application that converts voice recordings into text transcripts and generates concise summaries using custom-trained machine learning models. Built with a focus on privacy and efficiency, all processing happens locally with CPU-optimized models.

## 🚀 Key Features

- **Voice Recording**: Capture audio directly from your microphone
- **Live Transcription**: See real-time text as you speak
- **AI Summarization**: Get concise summaries of your notes
- **Privacy First**: All processing happens locally on your device
- **Lightweight**: Optimized for CPU-only execution
- **Modern UI**: Clean, responsive interface built with React and Tailwind CSS

## 🏗️ Architecture

```
├── src/                    # Frontend (React, TypeScript, Vite)
│   ├── components/         # Reusable UI components
│   ├── pages/             # Application pages
│   ├── hooks/             # Custom React hooks
│   └── types/             # TypeScript type definitions
│
├── backend/                # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── asr_handler.py     # ASR transcription handler
│   ├── summarizer_handler.py # Summarization handler
│   └── secret_models/     # Custom ML models
│
├── checkpoints/            # Trained model checkpoints
│   ├── asr/               # ASR model files
│   └── summarizer/        # Summarizer model files
│
└── model_cache/            # Cached models
```

## 🧠 Models

### ASR Model (Automatic Speech Recognition)
- **Architecture**: Custom CNN + BiLSTM + CTC
- Converts audio to text in real-time
- Optimized for CPU execution

### Summarizer Model (Text Summarization)
- **Architecture**: Transformer Encoder-Decoder
- Generates concise summaries from transcripts
- Character-level processing for accuracy

## 🛠️ Tech Stack

- **Frontend**: React, TypeScript, Vite, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI (Python)
- **ML Frameworks**: PyTorch, Librosa
- **Deployment**: Supabase (model storage)

## 📋 Requirements

- Node.js 16+
- Python 3.8+
- Modern web browser with microphone access
- PyTorch (CPU-only version)

## ⚙️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sunoy2004/SmartNote.git
   cd SmartNote
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start the development server**:
   ```bash
   # Terminal 1: Start backend
   cd backend
   python main.py
   
   # Terminal 2: Start frontend
   npm run dev
   ```

## 🔄 Workflow

1. Open the application in your browser
2. Click "Start Recording" to begin capturing audio
3. Speak naturally - see live transcription appear
4. Click "Stop Recording" when finished
5. View your transcript and AI-generated summary
6. Save or export your notes

## 🔧 API Endpoints

- `POST /voice-to-notes` - Complete pipeline (audio → transcript → summary)
- `POST /transcribe` - Transcribe audio file
- `POST /summarize` - Summarize text
- `GET /health` - Health check

## 📱 Browser Support

SmartNote works on all modern browsers that support:
- MediaRecorder API
- WebRTC
- WebSocket connections

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14.1+
- Edge 90+

## 🔒 Privacy & Security

- All audio processing happens locally
- No data is sent to external servers
- Models are cached locally after first download
- Microphone access is only used during recording

## 🚀 Deployment

### Local Deployment
1. Follow the setup instructions above
2. Build the frontend: `npm run build`
3. Serve the built files with any static server

### Netlify Deployment
1. Build the application: `npm run build`
2. This creates a `dist` folder with all production assets
3. Deploy the `dist` folder to Netlify using their dashboard or CLI
4. Set environment variables in Netlify dashboard if needed

### Cloud Deployment
1. Deploy backend to any cloud platform (Heroku, Railway, AWS, etc.)
2. Set environment variables for configuration
3. Serve frontend as static files
4. For Netlify: Point to the `dist` directory as the publish directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Vite](https://vitejs.dev/)
- UI components from [shadcn/ui](https://ui.shadcn.com/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Powered by [PyTorch](https://pytorch.org/)
- Audio processing with [Librosa](https://librosa.org/)
